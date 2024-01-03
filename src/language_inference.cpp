#include "language_inference.h"
#include "common/common.h"
#include "llama.h"
#include <atomic>
#include <cmath>
#include <cstdint>
#include <godot_cpp/classes/audio_server.hpp>
#include <godot_cpp/classes/project_settings.hpp>
#include <godot_cpp/classes/time.hpp>
#include <godot_cpp/core/error_macros.hpp>
#include <godot_cpp/core/memory.hpp>
#include <godot_cpp/variant/packed_vector2_array.hpp>
#include <godot_cpp/variant/utility_functions.hpp>
#include <string>
#include <vector>

TextToText *TextToText::singleton = nullptr;

TextToText *TextToText::get_singleton() {
	return singleton;
}

TextToText::TextToText() {
	singleton = this;
}

void TextToText::start_listen() {
	if (is_running == false) {
		is_running = true;
		worker.start(Callable(this, StringName("run")), Thread::Priority::PRIORITY_NORMAL);
		t_last_iter = Time::get_singleton()->get_ticks_msec();
	}
}

void TextToText::stop_inference() {
	is_running = false;
	if (worker.is_started()) {
		worker.wait_to_finish();
	}
}

void TextToText::load_model() {
	llama_free(context_instance);
	if (model.is_null()) {
		return;
	}
	PackedByteArray data = model->get_content();
	if (data.is_empty()) {
		return;
	}
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 99;
	language_model = llama_load_model_from_file(model->get_file().utf8().get_data(), model_params);
	// FIXME: fire 2024-01-03
	// params.prompt = "Hello my name is";
	llama_backend_init(OS::get_singleton()->get_processor_count());

	context_parameters = llama_context_default_params();
	// FIXME: fire 2024-01-03
	// context_parameters.seed  = 1234;
	// context_parameters.n_ctx = 2048;
	// context_parameters.n_threads = params.n_threads;
	// context_parameters.n_threads_batch = params.n_threads_batch == -1 ? params.n_threads : params.n_threads_batch;

	context_instance = llama_new_context_with_model(language_model, context_parameters);
	UtilityFunctions::print(llama_print_system_info());
}

TextToText::~TextToText() {
	singleton = nullptr;
	stop_inference();
	llama_free(context_instance);
}

void TextToText::add_string(const String buffer) {
	s_mutex.lock();
	const int total_length_of_sequence_with_prompt = 32;

	int32_t n_max_tokens = 512;
	std::vector<llama_token> tokens_list(n_max_tokens);
	int32_t n_tokens = llama_tokenize(language_model, buffer.utf8().get_data(), buffer.utf8().size(), tokens_list.data(), n_max_tokens, false /* no BOS */, false /* not special */);

	if (n_tokens < 0) {
		s_mutex.unlock();
		ERR_PRINT("Error: Tokenization failed due to insufficient token space.");
		return;
	}

	tokens_list.resize(n_tokens); // Adjust the size based on the actual number of tokens.

	llama_batch new_inference_batch = llama_batch_init(n_tokens, 0, 1);

	for (int i = 0; i < n_tokens; ++i) {
		llama_batch_add(batch, tokens_list[i], i, { 0 }, false);
	}

	new_inference_batch.logits[batch.n_tokens - 1] = true;

	if (llama_decode(context_instance, batch)) {
		s_mutex.unlock();
		ERR_PRINT("Error: Model decoding failed.");
		return;
	}

	int n_cur = batch.n_tokens;
	int n_decode = 0;

	const auto t_main_start = ggml_time_us();

	while (n_cur <= total_length_of_sequence_with_prompt) {
		auto n_vocab = llama_n_vocab(language_model);
		auto *logits = llama_get_logits_ith(context_instance, batch.n_tokens - 1);

		std::vector<llama_token_data> candidates;
		candidates.reserve(n_vocab);

		for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
			candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
		}

		llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

		const llama_token new_token_id = llama_sample_token_greedy(context_instance, &candidates_p);

		if (new_token_id == llama_token_eos(language_model) || n_cur == n_len) {
			UtilityFunctions::print(String());
			break;
		}
		Vector<char> token_piece;
		token_piece.resize(128);
		token_piece.fill(0);
		llama_token_to_piece(language_model, new_token_id, token_piece.ptrw(), token_piece.size());
		UtilityFunctions::print(String(token_piece.ptr()));

		llama_batch_clear(batch);

		llama_batch_add(batch, new_token_id, n_cur, { 0 }, true);

		n_decode += 1;
		n_cur += 1;

		if (llama_decode(context_instance, batch)) {
			s_mutex.unlock();
			ERR_PRINT(vformat("%s : failed to eval, return code %d", __func__, 1));
			return;
		}
	}

	const int64_t t_main_end = ggml_time_us();

	UtilityFunctions::print(vformat("%s: decoded %d tokens in %.2f s, speed: %.2f t/s",
			__func__, n_decode, (t_main_end - t_main_start) / 1000000.0f,
			n_decode / ((t_main_end - t_main_start) / 1000000.0f)));

	llama_print_timings(context_instance);

	s_mutex.unlock();
}

/** Get newly transcribed text. */
std::vector<transcribed_msg> TextToText::get_transcribed() {
	std::vector<transcribed_msg> transcribed;
	s_mutex.lock();
	transcribed = std::move(s_transcribed_msgs);
	s_transcribed_msgs.clear();
	s_mutex.unlock();
	return transcribed;
}

/** Run Whisper in its own thread to not block the main thread. */
void TextToText::run() {
	TextToText *speech_to_text_obj = TextToText::get_singleton();
	// whisper_full_params whisper_params = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
	// // See here for example https://github.com/ggerganov/whisper.cpp/blob/master/examples/stream/stream.cpp#L302
	// whisper_params.max_len = 1;
	// whisper_params.print_progress = false;
	// whisper_params.print_special = speech_to_text_obj->params.print_special;
	// whisper_params.print_realtime = false;
	// whisper_params.duration_ms = speech_to_text_obj->params.duration_ms;
	// whisper_params.print_timestamps = true;
	// whisper_params.translate = speech_to_text_obj->params.translate;
	// whisper_params.single_segment = true;
	// whisper_params.no_timestamps = false;
	// whisper_params.token_timestamps = true;
	// whisper_params.max_tokens = speech_to_text_obj->params.max_tokens;
	// whisper_params.language = speech_to_text_obj->params.language.c_str();
	// whisper_params.n_threads = speech_to_text_obj->params.n_threads;
	// whisper_params.speed_up = speech_to_text_obj->params.speed_up;
	// whisper_params.prompt_tokens = nullptr;
	// whisper_params.prompt_n_tokens = 0;
	// whisper_params.suppress_non_speech_tokens = true;
	// whisper_params.suppress_blank = true;
	// whisper_params.entropy_thold = speech_to_text_obj->params.entropy_threshold;
	// whisper_params.temperature = 0.0;
	// whisper_params.no_context = true;

	// /**
	//  * Experimental optimization: Reduce audio_ctx to 15s (half of the chunk
	//  * size whisper is designed for) to speed up 2x.
	//  * https://github.com/ggerganov/whisper.cpp/issues/137#issuecomment-1318412267
	//  */
	// whisper_params.audio_ctx = 768;

	// speech_to_text_obj->full_params = whisper_params;

	// /* When more than this amount of audio received, run an iteration. */
	// const int trigger_ms = 400;
	// const int n_samples_trigger = (trigger_ms / 1000.0) * WHISPER_SAMPLE_RATE;
	// /**
	//  * When more than this amount of audio accumulates in the audio buffer,
	//  * force finalize current audio context and clear the buffer. Note that
	//  * VAD may finalize an iteration earlier.
	//  */
	// // This is recommended to be smaller than the time wparams.audio_ctx
	// // represents so an iteration can fit in one chunk.
	// const int iter_threshold_ms = trigger_ms * 35;
	// const int n_samples_iter_threshold = (iter_threshold_ms / 1000.0) * WHISPER_SAMPLE_RATE;

	// /**
	//  * ### Reminders
	//  *
	//  * - Note that whisper designed to process audio in 30-second chunks, and
	//  *   the execution time of processing smaller chunks may not be shorter.
	//  * - The design of trigger and threshold allows inputing audio data at
	//  *   arbitrary rates with zero config. Inspired by Assembly.ai's
	//  *   real-time transcription API
	//  *   (https://github.com/misraturp/Real-time-transcription-from-microphone/blob/main/speech_recognition.py)
	//  */

	// /* VAD parameters */
	// // The most recent 3s.
	// const int vad_window_s = 3;
	// const int n_samples_vad_window = WHISPER_SAMPLE_RATE * vad_window_s;
	// // In VAD, compare the energy of the last 500ms to that of the total 3s.
	// const int vad_last_ms = 500;
	// // Keep the last 0.5s of an iteration to the next one for better
	// // transcription at begin/end.
	// const int n_samples_keep_iter = WHISPER_SAMPLE_RATE * 0.5;
	// const float vad_thold = 0.3f;
	// const float freq_thold = 200.0f;

	// /* Audio buffer */
	// std::vector<float> pcmf32;

	// /* Processing loop */
	// while (speech_to_text_obj->is_running) {
	// 	{
	// 		speech_to_text_obj->s_mutex.lock();
	// 		if (speech_to_text_obj->s_queued_pcmf32.size() < n_samples_trigger) {
	// 			speech_to_text_obj->s_mutex.unlock();
	// 			OS::get_singleton()->delay_msec(10);
	// 			continue;
	// 		}
	// 		speech_to_text_obj->s_mutex.unlock();
	// 	}
	// 	{
	// 		speech_to_text_obj->s_mutex.lock();
	// 		if (speech_to_text_obj->s_queued_pcmf32.size() > 2 * n_samples_iter_threshold) {
	// 			WARN_PRINT("Too much audio is going to be processed, result may not come out in real time");
	// 		}
	// 		speech_to_text_obj->s_mutex.unlock();
	// 	}
	// 	{
	// 		speech_to_text_obj->s_mutex.lock();
	// 		pcmf32.insert(pcmf32.end(), speech_to_text_obj->s_queued_pcmf32.begin(), speech_to_text_obj->s_queued_pcmf32.end());
	// 		speech_to_text_obj->s_queued_pcmf32.clear();
	// 		speech_to_text_obj->s_mutex.unlock();
	// 	}

	// 	if (!speech_to_text_obj->context_instance) {
	// 		ERR_PRINT("Context instance is null");
	// 		continue;
	// 	}
	// 	{
	// 		int ret = whisper_full(speech_to_text_obj->context_instance, speech_to_text_obj->full_params, pcmf32.data(), pcmf32.size());
	// 		if (ret != 0) {
	// 			ERR_PRINT("Failed to process audio, returned " + rtos(ret));
	// 			continue;
	// 		}
	// 	}
	// 	{
	// 		transcribed_msg msg;
	// 		const int n_segments = whisper_full_n_segments(speech_to_text_obj->context_instance);
	// 		for (int i = 0; i < n_segments; ++i) {
	// 			const int n_tokens = whisper_full_n_tokens(speech_to_text_obj->context_instance, i);
	// 			for (int j = 0; j < n_tokens; j++) {
	// 				auto token = whisper_full_get_token_data(speech_to_text_obj->context_instance, i, j);
	// 				// Idea from https://github.com/yum-food/TaSTT/blob/dbb2f72792e2af3ff220313f84bf76a9a1ddbeb4/Scripts/transcribe_v2.py#L457C17-L462C25
	// 				if (token.p > 0.6 && token.plog < -0.5) {
	// 					continue;
	// 				}
	// 				if (token.plog < -1.0) {
	// 					continue;
	// 				}
	// 				auto text = whisper_full_get_token_text(speech_to_text_obj->context_instance, i, j);
	// 				msg.text += text;
	// 			}
	// 		}
	// 		/**
	// 		 * Simple VAD from the "stream" example in whisper.cpp
	// 		 * https://github.com/ggerganov/whisper.cpp/blob/231bebca7deaf32d268a8b207d15aa859e52dbbe/examples/stream/stream.cpp#L378
	// 		 */
	// 		bool speech_has_end = false;

	// 		/* Need enough accumulated audio to do VAD. */
	// 		if ((int)pcmf32.size() >= n_samples_vad_window) {
	// 			std::vector<float> pcmf32_window(pcmf32.end() - n_samples_vad_window, pcmf32.end());
	// 			speech_has_end = vad_simple(pcmf32_window, WHISPER_SAMPLE_RATE, vad_last_ms,
	// 					vad_thold, freq_thold, false);
	// 			if (speech_has_end)
	// 				printf("speech end detected\n");
	// 		}
	// 		/**
	// 		 * Clear audio buffer when the size exceeds iteration threshold or
	// 		 * speech end is detected.
	// 		 */
	// 		if (pcmf32.size() > n_samples_iter_threshold || speech_has_end) {
	// 			const auto t_now = Time::get_singleton()->get_ticks_msec();
	// 			const auto t_diff = t_now - speech_to_text_obj->t_last_iter;
	// 			speech_to_text_obj->t_last_iter = t_now;

	// 			msg.is_partial = false;
	// 			/**
	// 			 * Keep the last few samples in the audio buffer, so the next
	// 			 * iteration has a smoother start.
	// 			 */
	// 			std::vector<float> last(pcmf32.end() - n_samples_keep_iter, pcmf32.end());
	// 			pcmf32 = std::move(last);
	// 		} else {
	// 			msg.is_partial = true;
	// 		}

	// 		speech_to_text_obj->s_mutex.lock();
	// 		s_transcribed_msgs.insert(s_transcribed_msgs.end(), std::move(msg));

	// 		std::vector<transcribed_msg> transcribed;
	// 		transcribed = std::move(s_transcribed_msgs);
	// 		s_transcribed_msgs.clear();

	// 		Array ret;
	// 		for (int i = 0; i < transcribed.size(); i++) {
	// 			Dictionary cur_transcribed_msg;
	// 			cur_transcribed_msg["is_partial"] = transcribed[i].is_partial;
	// 			String cur_text;
	// 			cur_transcribed_msg["text"] = cur_text.utf8(transcribed[i].text.c_str());
	// 			ret.push_back(cur_transcribed_msg);
	// 		};
	// 		speech_to_text_obj->call_deferred("emit_signal", "update_transcribed_msgs", ret);
	// 		speech_to_text_obj->s_mutex.unlock();
	// 	}
	// }
}

void TextToText::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_string", "buffer"), &TextToText::add_string);
	ClassDB::bind_method(D_METHOD("get_language_model"), &TextToText::get_language_model);
	ClassDB::bind_method(D_METHOD("set_language_model", "model"), &TextToText::set_language_model);
	ClassDB::bind_method(D_METHOD("start_listen"), &TextToText::start_listen);
	ClassDB::bind_method(D_METHOD("run"), &TextToText::run);
	ClassDB::bind_method(D_METHOD("stop_inference"), &TextToText::stop_inference);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "language_model", PROPERTY_HINT_RESOURCE_TYPE, "LlamaResource"), "set_language_model", "get_language_model");

	ADD_SIGNAL(MethodInfo("update_transcribed_msgs", PropertyInfo(Variant::ARRAY, "transcribed_msgs")));

	BIND_CONSTANT(SPEECH_SETTING_SAMPLE_RATE);
}
