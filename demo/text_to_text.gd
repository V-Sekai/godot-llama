extends Node

func _ready():
	var model = preload("res://addons/godot_llama/models/Phi-3-medium-128k-instruct-Q8_0.tres")
	TextToText.language_model = model
	TextToText.add_string("What can fly?")
	print("end _ready")
