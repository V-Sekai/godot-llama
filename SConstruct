#!/usr/bin/env python
import os
import sys

env = SConscript("thirdparty/godot-cpp/SConstruct")

env.Append(
    CPPDEFINES=[
        "HAVE_CONFIG_H",
        "PACKAGE=",
        "VERSION=",
        "CPU_CLIPS_POSITIVE=0",
        "CPU_CLIPS_NEGATIVE=0",
        "WHISPER_BUILD",
        "GGML_BUILD",
    ]
)

env.Prepend(CPPPATH=["thirdparty", "include", "thirdparty/llama.cpp/include", "thirdparty/llama.cpp/ggml/include"])
env.Append(CPPPATH=["src"])
env.Append(CPPDEFINES=['WHISPER_SHARED', 'GGML_SHARED'])
sources = [Glob("src/*.cpp")]
sources.extend([
    "thirdparty/llama.cpp/src/llama.cpp",
    "thirdparty/llama.cpp/src/unicode.cpp",
    "thirdparty/llama.cpp/src/unicode-data.cpp",
    "thirdparty/llama.cpp/common/common.cpp",
    "thirdparty/llama.cpp/common/grammar-parser.cpp",
    "thirdparty/llama.cpp/common/json-schema-to-grammar.cpp",
    "thirdparty/llama.cpp/common/console.cpp",
    "thirdparty/llama.cpp/common/ngram-cache.cpp",
    "thirdparty/llama.cpp/common/sampling.cpp",
    "thirdparty/llama.cpp/common/train.cpp",
    "thirdparty/llama.cpp/ggml/src/ggml-alloc.c",
    "thirdparty/llama.cpp/ggml/src/ggml-backend.c",
    "thirdparty/llama.cpp/ggml/src/ggml.c",
    "thirdparty/llama.cpp/ggml/src/ggml-quants.c",
])

if env["platform"] == "windows":
    env.Append(CPPFLAGS=["/EHsc"])
else:
    env.Append(CPPFLAGS=["-fexceptions"])

if env["platform"] == "macos" or env["platform"] == "ios":
    env.Append(LINKFLAGS=["-framework"])
    env.Append(LINKFLAGS=["Foundation"])
    env.Append(LINKFLAGS=["-framework"])
    env.Append(LINKFLAGS=["Metal"])
    env.Append(LINKFLAGS=["-framework"])
    env.Append(LINKFLAGS=["MetalKit"])
    env.Append(LINKFLAGS=["-framework"])
    env.Append(LINKFLAGS=["Accelerate"])
    env.Append(
        CPPDEFINES=[
            "GGML_USE_METAL",
            # Debug logs
            "GGML_METAL_NDEBUG",
            "GGML_USE_ACCELERATE"
        ]
    )
    sources.extend([
        Glob("thirdparty/llama.cpp/ggml/src/ggml-metal.m"),
    ])
else:
    sources.extend([
        "thirdparty/llama.cpp/ggml/src/ggml-vulkan.cpp",
        "thirdparty/volk/volk.c",
    ])
    env.Append(CPPPATH=["thirdparty/Vulkan-Headers/include", "thirdparty/volk"])
    env.Append(
        CPPDEFINES=[
        "GGML_USE_VULKAN",
        ]
    )

    
if env["platform"] == "macos":
	library = env.SharedLibrary(
		"bin/addons/godot_llama/bin/libgodot_llama{}.framework/libgodot_llama{}".format(
			env["suffix"], env["suffix"]
		),
		source=sources,
	)
else:
	library = env.SharedLibrary(
		"bin/addons/godot_llama/bin/libgodot_llama{}{}".format(env["suffix"], env["SHLIBSUFFIX"]),
		source=sources,
	)
Default(library)
