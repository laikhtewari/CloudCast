# General
import random
import sys
import os
import json
from time import time

# Inference libraries
import tflite_runtime.interpreter as tflite
import numpy as np

# Preprocessing libraries
from PIL import Image

BASE = "/home/mendel/demo-server/"
IMAGE_DIR = BASE + "test_images/"
PVLOG = BASE + "pv_log_test.npy"
MODEL = BASE + "model10.tflite"

def inference(img_id):
	inference_start = time()
	image = Image.open(IMAGE_DIR + f'image{int(img_id):04d}.png')
	x_train_2 = data_pv = np.load("pv_log_test.npy", allow_pickle=True)

	x_train_raw = image.getdata()
	x_train_raw = np.array(x_train_raw, dtype=np.float32)
	x_train_raw = np.reshape(x_train_raw, (64, 64, -1))
	x_train_raw = (1 - x_train_raw) * 255

	# ## Show example image
	# imageRGB = cv2.cvtColor(x_train_raw, cv2.COLOR_BGR2RGB) ## comment this line out if using png images that were correctly processed (i.e., correct colors)
	# image = Image.fromarray((imageRGB.astype(np.uint8)))
	# plt.imshow(image)
	# plt.show()

	# For the sake of demoing with one PNG image, repeat the same image 16 times, since model requires 16-image batches
	x_train_raw = np.repeat(x_train_raw[np.newaxis, :, :, :], 16, axis=0)[np.newaxis, :, :, :, :]

	x_train = x_train_raw.reshape(x_train_raw.shape[0], 64,64,48)
	norm = np.linalg.norm(x_train)
	x_train = x_train/norm

	x_train = np.float32(x_train)
	x_train_2 = np.float32(x_train_2)

	# print(x_train_raw.shape)
	# print(x_train_2.shape)

	interpreter = tflite.Interpreter(model_path=MODEL)
	interpreter.allocate_tensors()
	# Get input and output tensors.
	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	# print('Expected shape 0', input_details[0]['shape'])
	# print('Expected shape 1', input_details[1]['shape'])

	input_img = np.expand_dims(x_train[0], axis=0)
	input_pv = np.expand_dims(x_train_2[0], axis=0)

	# print('Actual shape 0', input_img.shape)
	# print('Actual shape 1', input_pv.shape)

	interpreter.set_tensor(input_details[0]['index'], input_img)
	# interpreter.set_tensor(input_details[1]['index'], input_pv)
	# ********HACK*********
	interpreter.set_tensor(input_details[1]['index'], np.repeat(input_pv[:, np.newaxis], 16, axis=1))

	model_start = time()
	interpreter.invoke()
	model_end = time()

	# The function `get_tensor()` returns a copy of the tensor data.
	# Use `tensor()` in order to get a pointer to the tensor.
	prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
	inference_end = time()

	model_latency = model_end - model_start
	inference_latency = inference_end - inference_start
	return (prediction, model_latency, inference_latency) # pred, latency

def num_images():
	return len(os.listdir(IMAGE_DIR))

if __name__ == "__main__":
	try:
		IMAGE_ID = sys.argv[1]
		out = {}
		pv_pred, model_latency, rt_latency = inference(IMAGE_ID)
		out["pv_pred"] = f"{pv_pred:.4f}"
		out["model_latency"] = f"{model_latency * 1000:.2f}"
		out["rt_latency"] = f"{rt_latency * 1000:.2f}"
	except Exception as e:
		out["error"] = repr(e)
	finally:
		print(json.dumps(out))
		sys.stdout.flush()

