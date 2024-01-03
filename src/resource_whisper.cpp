#include "resource_whisper.h"
#include <iostream>

#include <godot_cpp/classes/file_access.hpp>

PackedByteArray LlamaResource::get_content() {
	PackedByteArray content;
	String p_path = get_file();
	content = FileAccess::get_file_as_bytes(p_path);
	return content;
}
