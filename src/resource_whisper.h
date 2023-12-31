#ifndef WHISPER_RESOURCE_H
#define WHISPER_RESOURCE_H

#include <godot_cpp/classes/resource.hpp>

using namespace godot;

class LlamaResource : public Resource {
	GDCLASS(LlamaResource, Resource);

protected:
	static void _bind_methods() {}
	String file;

public:
	void set_file(const String &p_file) {
		file = p_file;
		emit_changed();
	}

	String get_file() {
		return file;
	}

	PackedByteArray get_content();
	LlamaResource() {}
	~LlamaResource() {}
};
#endif // RESOURCE_JSON_H
