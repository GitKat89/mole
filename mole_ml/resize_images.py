import numpy as np
from PIL import Image
import os
import scipy.misc
import glob
import sys
import cv2

import random
import keras_preprocessing.image
from keras.preprocessing.image import save_img


from PIL import Image
import argparse

#read_directory = "/Users/deinemudda/Desktop/ML/mole/mole_ml/raw_data/ben/"
#save_directory = "/Users/deinemudda/Desktop/ML/mole/mole_ml/raw_data/ben_resized/"

#infilename = "/Users/deinemudda/Desktop/ML/mole/mole_ml/raw_data/ben/ISIC_0000020_downsampled.jpg"
#outfilename = "./resized_image.jpg"

def load_image( infilename ) :
    img = Image.open( infilename )
    img.load()
    data = np.asarray( img, dtype="int32" )
    return data

def save_image( npdata, outfilename ) :
    img = Image.fromarray( np.asarray( np.clip(npdata,0,255), dtype="uint8"), "L" )
    img.save( outfilename )
    # read image
    #img = load_image(infilename)

def make_images_squared(img):
    ht, wd, cc= img.shape
    # create new image of desired size and color (black) for padding
    ww = 224
    hh = 224
    color = (0,0,0)
    result = np.full((hh,ww,cc), color, dtype=np.uint8)

    # compute center offset
    xx = (ww - wd) // 2
    yy = (hh - ht) // 2

    # copy img image into center of result image
    result[yy:yy+ht, xx:xx+wd] = img
    return result

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def save_resized_image(img, new_filename, size):
    img_res = cv2.resize(img, dsize=size)
    cv2.imwrite(new_filename, cv2.cvtColor(img_res, cv2.COLOR_RGB2BGR))


def load_and_crop_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Wraps keras_preprocessing.image.utils.loag_img() and adds cropping.
    Cropping method enumarated in interpolation
    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rgb", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation and crop methods used to resample and crop the image
            if the target size is different from that of the loaded image.
            Methods are delimited by ":" where first part is interpolation and second is crop
            e.g. "lanczos:random".
            Supported interpolation methods are "nearest", "bilinear", "bicubic", "lanczos",
            "box", "hamming" By default, "nearest" is used.
            Supported crop methods are "none", "center", "random".
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """

    # Decode interpolation string. Allowed Crop methods: none, center, random
    interpolation, crop = interpolation.split(":") if ":" in interpolation else (interpolation, "none")  

    if crop == "none":
        return keras_preprocessing.image.utils.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=target_size,
                                            interpolation=interpolation)

    # Load original size image using Keras
    img = keras_preprocessing.image.utils.load_img(path, 
                                            grayscale=grayscale, 
                                            color_mode=color_mode, 
                                            target_size=None, 
                                            interpolation=interpolation)

    # Crop fraction of total image
    crop_fraction = 1 #0.875
    target_width = target_size[1]
    target_height = target_size[0]

    if target_size is not None:        
        if img.size != (target_width, target_height):

            if crop not in ["center", "random"]:
                raise ValueError('Invalid crop method {} specified.', crop)

            if interpolation not in keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(interpolation,
                        ", ".join(keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS.keys())))
            
            resample = keras_preprocessing.image.utils._PIL_INTERPOLATION_METHODS[interpolation]

            width, height = img.size

            # Resize keeping aspect ratio
            # result shold be no smaller than the targer size, include crop fraction overhead
            target_size_before_crop = (target_width/crop_fraction, target_height/crop_fraction)
            ratio = max(target_size_before_crop[0] / width, target_size_before_crop[1] / height)
            target_size_before_crop_keep_ratio = int(width * ratio), int(height * ratio)
            img = img.resize(target_size_before_crop_keep_ratio, resample=resample)

            width, height = img.size

            if crop == "center":
                left_corner = int(round(width/2)) - int(round(target_width/2))
                top_corner = int(round(height/2)) - int(round(target_height/2))
                return img.crop((left_corner, top_corner, left_corner + target_width, top_corner + target_height))
            elif crop == "random":
                left_shift = random.randint(0, int((width - target_width)))
                down_shift = random.randint(0, int((height - target_height)))
                return img.crop((left_shift, down_shift, target_width + left_shift, target_height + down_shift))

    return img

def main(read_directory, save_directory, make_squared, size):

    create_folder(save_directory)

    for filename in os.listdir(read_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            #image = load_image(os.path.join(read_directory, filename))
            #if make_images_squared:
            #    image = make_images_squared(image)
            path =  os.path.join(read_directory, filename)
            image = load_and_crop_img(path, target_size=(224,224), interpolation='nearest:center')
            #save_resized_image(image, os.path.join(save_directory, filename), size)
            target_path = os.path.join(save_directory, filename)
            save_img(target_path, image)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument("--read_directory", "-rd", help="", type=str)
    parser.add_argument("--save_directory", "-sd", help="", type=str)
    parser.add_argument("--make_squared", "-sq", help="if True: The images are first squared and then resized", type=bool, default = False)
    parser.add_argument("--size", "-s", help="", type=tuple, default = (224,224))
   
    args = parser.parse_args()

    read_directory = args.read_directory
    print('read_directory:', read_directory)
    
    save_directory = args.save_directory
    print('save_directory:', save_directory)

    make_squared = args.make_squared
    print('make_squared:', make_squared)

    size = args.size
    print('size:', size)

    main(read_directory, save_directory, make_squared, size)

