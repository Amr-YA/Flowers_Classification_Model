import argparse
from ast import arg
from unicodedata import category

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import sys

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description="predict a flower class from its JPG image")

parser.add_argument("input", action="store", default="./test_images", type=str, help="Path to the folder with images to be classified")
parser.add_argument("model", action="store", default="best_model.h5", type=str, help="Path H5 model to be used for prediction")
parser.add_argument("-k", "--top_k", action="store", default=5, type=int, help="Return the top K most likely classes")
parser.add_argument("-c", "--category_names", action="store", default="./label_map.json", type=str, help="Path to a JSON file mapping labels to flower names")
args = parser.parse_args()

# loaded_model = tf.keras.models.load_model(args.model, custom_objects={'KerasLayer': hub.KerasLayer})
model_path = args.model
image_path = args.input
label_file = args.category_names
top_k = args.top_k

## handling error in loading the json file
try:
    with open(label_file, 'r') as f:
        class_names = json.load(f)
    # print("Loaded the mapping JSON file")
except Exception as e:
    try:
        with open("./label_map.json", 'r') as f:
            class_names = json.load(f)
        print("- Error loading the specified mapping JSON file, default file will be used")
    except Exception as e:
        print("- Error loading the specified or default mapping JSON file, please ensure there's a valid JSON file for the flower name mapping")
        sys.exit("Error")

## handling errors in loading model
try:
    loaded_model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    # print("Loaded the prediction model")
except Exception as e:
    try:
        loaded_model = tf.keras.models.load_model("best_model.h5", custom_objects={'KerasLayer': hub.KerasLayer})
        print("- Error loading the specified prediction model, default model will be used")
    except Exception as e:
        print("- Error loading the specified or default prediction model, please ensure there's a valid model.h5 exist")
        sys.exit("Error")

def process_image(image):
    image_size = 224
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image

def predict():
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image, axis=0)
    predictions = loaded_model.predict(processed_image)
    probs, labels = tf.nn.top_k(predictions[0], top_k)
    classes = [class_names[str(c+1)] for c in labels.numpy()]
    return probs, labels, classes
print(f"Predictions for file: {image_path}:")
probs, labels, classes = predict()

for prob, label, cls in zip(probs, labels, classes) :
    print(f"Prediction: {cls}, with accuracy of: {prob*100:0.2f}%")

