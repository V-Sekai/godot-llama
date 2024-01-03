#include "resource_loader_whisper.h"
#include "resource_whisper.h"

Variant ResourceFormatLoaderLlama::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<LlamaResource> whisper_model = memnew(LlamaResource);
	whisper_model->set_file(p_path);
	return whisper_model;
}
PackedStringArray ResourceFormatLoaderLlama::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("bin");
	return array;
}
bool ResourceFormatLoaderLlama::_handles_type(const StringName &type) const {
	return ClassDB::is_parent_class(type, "LlamaResource");
}
String ResourceFormatLoaderLlama::_get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "bin") {
		return "LlamaResource";
	}
	return String();
}
