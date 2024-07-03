#include "resource_loader_gguf.h"
#include "resource_gguf.h"

Variant ResourceFormatLoaderLlama::_load(const String &p_path, const String &original_path, bool use_sub_threads, int32_t cache_mode) const {
	Ref<GGUFResource> whisper_model = memnew(GGUFResource);
	whisper_model->set_file(p_path);
	return whisper_model;
}
PackedStringArray ResourceFormatLoaderLlama::_get_recognized_extensions() const {
	PackedStringArray array;
	array.push_back("bin");
	return array;
}
bool ResourceFormatLoaderLlama::_handles_type(const StringName &type) const {
	return ClassDB::is_parent_class(type, "GGUFResource");
}
String ResourceFormatLoaderLlama::_get_resource_type(const String &p_path) const {
	String el = p_path.get_extension().to_lower();
	if (el == "bin") {
		return "GGUFResource";
	}
	return String();
}
