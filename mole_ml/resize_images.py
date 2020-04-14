import numpy as np
from PIL import Image
import os
import scipy.misc
import glob
import sys
import cv2


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
    # create new image of desired size and color (blue) for padding
    ww = 1024
    hh = 1024
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

def main(read_directory, save_directory, make_squared, size):

    create_folder(save_directory)

    for filename in os.listdir(read_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image = load_image(os.path.join(read_directory, filename))
            if make_images_squared:
                image = make_images_squared(image)
            save_resized_image(image, os.path.join(save_directory, filename), size)

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

