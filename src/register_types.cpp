#include "register_types.h"

#include "resource_loader_whisper.h"
#include "resource_whisper.h"
#include "language_inference.h"

#include <godot_cpp/classes/resource_loader.hpp>

static Ref<ResourceFormatLoaderLlama> llama_loader;

static SpeechToText *SpeechToTextPtr;

void initialize_llama_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_CLASS(SpeechToText);
	GDREGISTER_CLASS(LlamaResource);
	GDREGISTER_CLASS(ResourceFormatLoaderLlama);
	llama_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(llama_loader);

	SpeechToTextPtr = memnew(SpeechToText);
	Engine::get_singleton()->register_singleton("SpeechToText", SpeechToText::get_singleton());
}

void uninitialize_llama_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Engine::get_singleton()->unregister_singleton("SpeechToText");
	memdelete(SpeechToTextPtr);

	ResourceLoader::get_singleton()->remove_resource_format_loader(llama_loader);
	llama_loader.unref();
}

extern "C" {

GDExtensionBool GDE_EXPORT godot_llama_library_init(const GDExtensionInterfaceGetProcAddress p_get_proc_address, GDExtensionClassLibraryPtr p_library, GDExtensionInitialization *r_initialization) {
	godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);

	init_obj.register_initializer(initialize_llama_module);
	init_obj.register_terminator(uninitialize_llama_module);
	init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

	return init_obj.init();
}
}
