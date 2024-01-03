#include "register_types.h"

#include "language_inference.h"
#include "resource_loader_whisper.h"
#include "resource_whisper.h"

#include <godot_cpp/classes/resource_loader.hpp>

int LLAMA_BUILD_NUMBER = 80;
char const *LLAMA_COMMIT = "b399466";
char const *LLAMA_COMPILER = "";
char const *LLAMA_BUILD_TARGET = "unknown";

static Ref<ResourceFormatLoaderLlama> llama_loader;

static TextToText *TextToTextPtr;

void initialize_llama_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	GDREGISTER_CLASS(TextToText);
	GDREGISTER_CLASS(LlamaResource);
	GDREGISTER_CLASS(ResourceFormatLoaderLlama);
	llama_loader.instantiate();
	ResourceLoader::get_singleton()->add_resource_format_loader(llama_loader);

	TextToTextPtr = memnew(TextToText);
	Engine::get_singleton()->register_singleton("TextToText", TextToText::get_singleton());
}

void uninitialize_llama_module(ModuleInitializationLevel p_level) {
	if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
		return;
	}
	Engine::get_singleton()->unregister_singleton("TextToText");
	memdelete(TextToTextPtr);

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
