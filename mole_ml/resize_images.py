import numpy as np
from PIL import Image
import os
import scipy.misc
import glob
import sys

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

def resize_image(img):
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

def save_resized_image(img, new_filename):
    scipy.misc.toimage(img, cmin=0.0, cmax=...).save(new_filename)
# save result
#save_image(result, outfilename)

#for filepath in glob.iglob('read_directory/*.jpg'):
#    print(filepath)
def main(read_directory, save_directory):
    for filename in os.listdir(read_directory):
        if filename.endswith(".jpg") or filename.endswith(".jpeg"):
            print("filename:   ", read_directory + filename)
            loaded_image = load_image(read_directory + filename)
            resized_image = resize_image(loaded_image)
            save_resized_image(resized_image, save_directory+ "/" + filename)


if __name__ == '__main__':
    read_directory = sys.argv[1]
    print('read_directory:', read_directory)
    
    save_directory = sys.argv[2]
    print('save_directory:', sys.argv[2])
    main(read_directory, save_directory)

