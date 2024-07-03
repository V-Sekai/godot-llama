#include "resource_gguf.h"
#include <iostream>

#include <godot_cpp/classes/file_access.hpp>

PackedByteArray GGUFResource::get_content() {
	PackedByteArray content;
	String p_path = get_file();
	content = FileAccess::get_file_as_bytes(p_path);
	return content;
}

void GGUFResource::_bind_methods() {
	ClassDB::bind_method(D_METHOD("set_file", "file"), &GGUFResource::set_file);
	ClassDB::bind_method(D_METHOD("get_file"), &GGUFResource::get_file);
	ADD_PROPERTY(PropertyInfo(Variant::STRING, "file"), "set_file", "get_file");
}
