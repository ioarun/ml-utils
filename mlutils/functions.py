import os
import torch
import tensorflow as tf
import onnx
from onnx2keras import onnx_to_keras

def pytorch2onnx(model, sample_data, target_path):
	# Export the model
	input_x = sample_data.reshape(1, sample_data.shape[1]) 
	output = model(input_x)
	torch.onnx.export(model, input_x, target_path, export_params=True, input_names=['main_input'], output_names=['main_output'])
	print ("Exported pytorch model to ONNX and saved it as {}".format(target_path))

def onnx2keras(model, model_path, target_path):
	onnx_model = onnx.load(model_path)
	k_model = onnx_to_keras(onnx_model, ['main_input'])
	k_model.save(target_path)
	print ("Exported ONNX model to keras model and saved it as {}".format(target_path))

def keras2tflite(model, model_path, target_path):
    # Load the tensorflow model
    model = tf.keras.models.load_model(model_path)
    # TFlite model
    # converter = tf.lite.TFLiteConverter.from_keras_model(model) # TF 2.x
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_path) # TF 1.x
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    # Save the TF Lite model.
    with tf.io.gfile.GFile(target_path, 'wb') as f:
      f.write(tflite_model)
    print ("Exported keras model to tflite model and saved it as {}".format(target_path))


def tflite2cpp(model, model_path, target_path):
    os.system('xxd -i '+model_path+' > '+target_path+'')
    print ("Exported tflite model to c++ model and saved it as {}".format(target_path))

def check_onnx_model(model_path):
	onnx_model = onnx.load(model_path)
	print('The model is:\n{}'.format(onnx_model))
	# Check the model
	try:
		onnx.checker.check_model(onnx_model)
	except onnx.checker.ValidationError as e:
		print('The model is invalid: %s' % e)
	else:
		print('The model is valid!')