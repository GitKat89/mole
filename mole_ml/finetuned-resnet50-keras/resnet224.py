import os
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
np.random.seed(11) # It's my lucky number
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
import itertools
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.np_utils import to_categorical # convert to one-hot-encoding
from keras.optimizers import Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.resnet50 import ResNet50
from keras import backend as K 

def resize_images():

def preprocess():
    """load images and create labels """

    folder_benign_train = '../input/data/train/benign'
    folder_malignant_train = '../input/data/train/malignant'

    folder_benign_test = '../input/data/test/benign'
    folder_malignant_test = '../input/data/test/malignant'

    read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

    # Load in training pictures 
    ims_benign = [read(os.path.join(folder_benign_train, filename)) for filename in os.listdir(folder_benign_train)]
    X_benign = np.array(ims_benign, dtype='uint8')
    ims_malignant = [read(os.path.join(folder_malignant_train, filename)) for filename in os.listdir(folder_malignant_train)]
    X_malignant = np.array(ims_malignant, dtype='uint8')

    # Load in testing pictures
    ims_benign = [read(os.path.join(folder_benign_test, filename)) for filename in os.listdir(folder_benign_test)]
    X_benign_test = np.array(ims_benign, dtype='uint8')
    ims_malignant = [read(os.path.join(folder_malignant_test, filename)) for filename in os.listdir(folder_malignant_test)]
    X_malignant_test = np.array(ims_malignant, dtype='uint8')

    # Create labels
    y_benign = np.zeros(X_benign.shape[0])
    y_malignant = np.ones(X_malignant.shape[0])

    y_benign_test = np.zeros(X_benign_test.shape[0])
    y_malignant_test = np.ones(X_malignant_test.shape[0])

    # Merge data 
    X_train = np.concatenate((X_benign, X_malignant), axis = 0)
    y_train = np.concatenate((y_benign, y_malignant), axis = 0)

    X_test = np.concatenate((X_benign_test, X_malignant_test), axis = 0)
    y_test = np.concatenate((y_benign_test, y_malignant_test), axis = 0)

    # Shuffle data
    s = np.arange(X_train.shape[0])
    np.random.shuffle(s)
    X_train = X_train[s]
    y_train = y_train[s]

    s = np.arange(X_test.shape[0])
    np.random.shuffle(s)
    X_test = X_test[s]
    y_test = y_test[s]

    #one hot encode
    y_train = to_categorical(y_train, num_classes= 2)
    y_test = to_categorical(y_test, num_classes= 2)

    # Scale; With data augmentation to prevent overfitting 
    X_train = X_train/255.
    X_test = X_test/255.

    return X_train, X_test, y_train, y_test
def create_model():
    input_shape = (224,224,3)
    lr = 1e-5
    epochs = 50
    batch_size = 64

    model = ResNet50(include_top=True,
                    weights= None,
                    input_tensor=None,
                    input_shape=input_shape,
                    pooling='avg',
                    classes=2)

    model.compile(optimizer = Adam(lr) ,
                loss = "binary_crossentropy", 
                metrics=["accuracy"])

    history = model.fit(X_train, y_train, validation_split=0.2,
                        epochs= epochs, batch_size= batch_size, verbose=2, 
                        callbacks=[learning_rate_reduction]
                    )
    return history

def evaluate_model():

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument("--datapath", "-d", help="", type=str, default= '../data/')
    parser.add_argument("--batchsize", "-b", help="", type=int, default = 2)
    parser.add_argument("--epochs", "-e", help="", type=int, default = 2)
    #parser.add_argument("--cores", "-c", help="", type=str, default = '1')

    args = parser.parse_args()
    print("args: ", args)

    DATA_DIR =args.datapath
    BATCH_SIZE = args.batchsize
    n_epochs = args.epochs

    X_train, X_test, y_train, y_test = preprocess()
    history = create_model() # add parameters