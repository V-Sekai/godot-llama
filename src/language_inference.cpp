#include "language_inference.h"
#include "../thirdparty/llama.cpp/common/common.h"
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
	llama_backend_init();

	context_parameters = llama_context_default_params();
	context_parameters.seed = 1234;
	context_parameters.n_ctx = 4096;
	context_parameters.n_threads = OS::get_singleton()->get_processor_count();
	context_parameters.n_threads_batch = context_parameters.n_threads;

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
	const int32_t total_length_of_sequence_with_prompt = 512;
	std::vector<llama_token> tokens_list(total_length_of_sequence_with_prompt);
	const String prompt = "Hello my name is";
	int32_t n_tokens = llama_tokenize(language_model, prompt.utf8().get_data(), buffer.utf8().size(), tokens_list.data(), total_length_of_sequence_with_prompt, true, false);
	if (n_tokens < 0) {
		s_mutex.unlock();
		ERR_PRINT("Error: Tokenization failed due to insufficient token space.");
		return;
	}
	n_tokens = llama_tokenize(language_model, buffer.utf8().get_data(), buffer.utf8().size(), tokens_list.data(), total_length_of_sequence_with_prompt, false /* no BOS */, false /* not special */);
	if (n_tokens < 0) {
		s_mutex.unlock();
		ERR_PRINT("Error: Tokenization failed due to insufficient token space.");
		return;
	}

	tokens_list.resize(n_tokens); // Adjust the size based on the actual number of tokens.

	llama_batch new_inference_batch = llama_batch_init(n_tokens, 0, 1);

	for (int i = 0; i < n_tokens; ++i) {
		llama_batch_add(new_inference_batch, tokens_list[i], i, { 0 }, false);
	}

	new_inference_batch.logits[new_inference_batch.n_tokens - 1] = true;

	if (llama_decode(context_instance, new_inference_batch)) {
		s_mutex.unlock();
		ERR_PRINT("Error: Model decoding failed.");
		return;
	}

	int n_cur = new_inference_batch.n_tokens;
	int n_decode = 0;

	const auto t_main_start = ggml_time_us();

	Array ret;

	while (n_cur <= total_length_of_sequence_with_prompt) {
		auto n_vocab = llama_n_vocab(language_model);
		auto *logits = llama_get_logits_ith(context_instance, new_inference_batch.n_tokens - 1);

		std::vector<llama_token_data> candidates;
		candidates.reserve(n_vocab);

		for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
			candidates.emplace_back(llama_token_data{ token_id, logits[token_id], 0.0f });
		}

		llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

		const llama_token new_token_id = llama_sample_token_greedy(context_instance, &candidates_p);

		if (new_token_id == llama_token_eos(language_model) || n_cur == total_length_of_sequence_with_prompt) {
			UtilityFunctions::print(String());
			break;
		}
		Dictionary cur_transcribed_msg;
		Vector<char> token_piece;
		token_piece.resize(128);
		token_piece.fill(0);

		cur_transcribed_msg["text"] = String().utf8(llama_token_to_piece(context_instance, new_token_id).c_str());
		ret.push_back(cur_transcribed_msg);

		UtilityFunctions::print(String(token_piece.ptr()));

		llama_batch_clear(new_inference_batch);

		llama_batch_add(new_inference_batch, new_token_id, n_cur, { 0 }, true);

		n_decode += 1;
		n_cur += 1;

		if (llama_decode(context_instance, new_inference_batch)) {
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

	call_deferred("emit_signal", "update_transcribed_msgs", ret);

	s_mutex.unlock();
}

void TextToText::_bind_methods() {
	ClassDB::bind_method(D_METHOD("add_string", "buffer"), &TextToText::add_string);
	ClassDB::bind_method(D_METHOD("get_language_model"), &TextToText::get_language_model);
	ClassDB::bind_method(D_METHOD("set_language_model", "model"), &TextToText::set_language_model);
	ClassDB::bind_method(D_METHOD("start_listen"), &TextToText::start_listen);
	ClassDB::bind_method(D_METHOD("stop_inference"), &TextToText::stop_inference);
	ADD_PROPERTY(PropertyInfo(Variant::OBJECT, "language_model", PROPERTY_HINT_RESOURCE_TYPE, "LlamaResource"), "set_language_model", "get_language_model");

	ADD_SIGNAL(MethodInfo("update_transcribed_msgs", PropertyInfo(Variant::ARRAY, "transcribed_msgs")));

	BIND_CONSTANT(SPEECH_SETTING_SAMPLE_RATE);
}
