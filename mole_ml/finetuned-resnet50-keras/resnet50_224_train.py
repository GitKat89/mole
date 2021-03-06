import math, json, os, sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
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
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.applications.resnet50 import ResNet50
from keras import optimizers
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
import cv2

import argparse

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def preprocess():
    """load images and create labels """

    folder_benign_train = '../data/train/benign'
    folder_malignant_train = '../data/train/malignant'

    folder_benign_test = '../data/test/benign'
    folder_malignant_test = '../data/test/malignant'

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

    # one hot encoding
    y_train = to_categorical(y_train, num_classes= 2)
    y_test = to_categorical(y_test, num_classes= 2)

    # scaling from 0 to 255 to 0 to 1
    X_train = X_train/255.
    X_test = X_test/255.

    return X_train, X_test, y_train, y_test

def plot(history):

    with open('output/file.json', 'w') as f:
        json.dump(str(history.history), f)

    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('output/history_acc.png')

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('output/history_loss.png')

def create_model(dim_x, dim_y, lr):
    input_shape = (dim_x,dim_y,3)
   
    model = ResNet50(include_top=True,
                    weights= None,
                    input_tensor=None,
                    input_shape=input_shape,
                    pooling='avg',
                    classes=2)

    model.compile(optimizer = Adam(lr) ,
                loss = "categorical_crossentropy", 
                metrics=["accuracy"])
    
    model.summary()
    return model

def create_model_tl(dim_x, dim_y, lr):
    input_shape = (dim_x,dim_y,3)

    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    output_layer = resnet.layers[-1].output
    output_layer = keras.layers.Flatten()(output_layer)

    resnet = Model(resnet.input, output=output_layer)

    #for layer in resnet.layers:
    #    layer.trainable = False
    
    model.layers[0].trainable = False

    model = Sequential()
    model.add(resnet)
  
    model.add(Dense(1, activation='softmax', name="output"))
    sgd = optimizers.SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

    model.summary()
    return model

def read_test_labels():
    df = pd.read_csv('../ISIC_2019_Training_GroundTruth.csv')
    df_test = df[df.index % 101 == 0] #Selects every 101th row
    df = df.drop(df.columns[[2, 3, 4, 5, 6, 7, 8, 9]], axis = 1)
    return df

def test(data_dir, dim_x,dim_y):
    
    test_dir = os.path.join(data_dir, 'test')
    batch_size_test = 1
    test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    test_batches = test_gen.flow_from_directory(directory = test_dir, target_size = (dim_x, dim_y),
        batch_size = batch_size_test,
        class_mode = None,
        shuffle = False,
        seed = 123
    )

    model = load_model('output/resnet50_224_best.h5')
    test_batches.reset()
    pred = model.predict_generator(test_batches, steps = len(test_batches), verbose = 1)
    predicted_class_indices = np.argmax(pred, axis = 1)

    # initialize plot
    f, ax = plt.subplots(5, 5, figsize = (15, 15))
    
    df_labels = read_test_labels()
    df_result = pd.DataFrame(columns = ['image', 'labeled', 'predicted'])
    image_count = len(test_batches.filenames)
   
    for i in range(0, image_count):
        # read the image
        image_file_path = test_batches.filenames[i]
        image_name = os.path.splitext(os.path.basename(image_file_path))[0]

        predicted_class = predicted_class_indices[i]
        labeled_class = int(df_labels.loc[df_labels['image'] == image_name, 'MEL'].to_numpy()[0])

        df_result = df_result.append({'image': image_name, 'labeled': labeled_class,  'predicted':predicted_class}, ignore_index=True)
        
        # create image comparison for the first 25 images
        if i < 25:
            imgBGR = cv2.imread(os.path.join(test_dir, image_file_path))
            image_name = image_file_path.rsplit('.', 1)[0]

            imgRGB = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
            # a if condition else b
            predicted_class_text = "Malignant" if predicted_class else "Benign"
            
            labeled_class_text = "Malignant" if labeled_class else "Benign" 

            ax[i//5, i%5].imshow(imgRGB)
            ax[i//5, i%5].axis('off')
            ax[i//5, i%5].set_title("Predicted:{} \n Labeled:{}".format(predicted_class, labeled_class))   

    plt.savefig('output/test_predictions.png')
    df_result.to_csv('output/test_predictions.csv')
    #calculate statistics
    diff1 = df_result['predicted'] - df_result['labeled'] 
    diff2 = df_result['labeled'] - df_result['predicted']
    percentage_correctly_classified = (image_count + (diff1 * diff2).sum()) / image_count * 100
    print("{0:.2f} percent of test images has been correctly classified".format(percentage_correctly_classified))

def train(data_dir, epochs, batch_size, dim_x, dim_y, lr):

    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')

    create_folder('output')

    patience = 5

    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

    num_train_steps = math.floor(num_train_samples/batch_size)
    num_valid_steps = math.floor(num_valid_samples/batch_size)

    train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_batches = train_gen.flow_from_directory(train_dir,target_size=(dim_x, dim_y),batch_size=batch_size, class_mode='categorical')
    val_batches = val_gen.flow_from_directory(val_dir, target_size=(dim_x, dim_y) , batch_size=batch_size, class_mode='categorical')

    model = create_model(dim_x, dim_y, lr)

    # Set a learning rate annealer
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=patience, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=lr)

    early_stopping = EarlyStopping(patience=patience)
    checkpointer = ModelCheckpoint('output/resnet50_224_best.h5', verbose=1, save_best_only=True)

    classes = list(iter(train_batches.class_indices))

    #for c in train_batches.class_indices:
    #    print("Class indices: " + str(c))
    #    classes[train_batches.class_indices[c]] = c
    #model.classes = classes

    history = model.fit_generator(train_batches, steps_per_epoch=num_train_steps, epochs=epochs, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)

    model.load_weights("output/resnet50_224_best.h5")

    model.save('output/resnet50_224.h5')

    plot(history)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument("--datapath", "-d", help="", type=str, default= '../data/')
    parser.add_argument("--batchsize", "-b", help="", type=int, default = 2)
    parser.add_argument("--epochs", "-e", help="", type=int, default = 2)
    parser.add_argument("--test", "-t", help="Activates if a test run should be conducted.", type=bool, default= True)

    args = parser.parse_args()
    print("args: ", args)

    data_dir = args.datapath
    batch_size = args.batchsize
    epochs = args.epochs
    dim_x = 224
    dim_y = 224
    lr = 1e-5
 
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    create_folder('output')

    train(data_dir, epochs, batch_size, dim_x, dim_y, lr)
    
    if args.test:
        test(data_dir, dim_x, dim_y)

    