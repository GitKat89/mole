import IPython
import json, os, re, sys, time
import numpy as np
import cv2

from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

import resize_images

def predict(resized_img, model):
    #img = image.load_img(img_path, target_size=(224, 224))
    #x = image.img_to_array(img)
    print("shape of loaded image before resize: ", resized_img.shape)
    x = np.expand_dims(resized_img, axis=0)
    print("shape of loaded image after resize: ", x.shape)
    preds = model.predict(x)[0]
    print("preds: ", preds)
    return preds

def get_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (300, 300))
    img.astype(np.float32)
    img = img / 255.0
    return img

if __name__ == '__main__':
    model_path = sys.argv[1]
    print('Loading model:', model_path)
    t0 = time.time()
    model = load_model(model_path)
    t1 = time.time()
    print('Loaded in:', t1-t0)

    test_path = sys.argv[2]
    print('Generating predictions on image:', sys.argv[2])

    loaded_image = get_image(sys.argv[2])
    
    preds = predict(loaded_image, model)

    
