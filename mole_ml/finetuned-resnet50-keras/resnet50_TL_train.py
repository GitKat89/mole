import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Flatten
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from keras import optimizers

import argparse

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def create_model(input_shape):
    resnet = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)

    output_layer = resnet.layers[-1].output
    output_layer = keras.layers.Flatten()(output_layer)

    resnet = Model(resnet.input, output=output_layer)

    for layer in resnet.layers:
        layer.trainable = False

    model = Sequential()
    model.add(resnet)
    model.add(Dense(512, activation='relu', input_dim=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid', name="output"))
    
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5),metrics=['accuracy'])

    model.summary()
    return model


def main(train_dir, val_dir, epochs, batch_size, cores):
    DIM_X = 224
    DIM_Y = 224
    SHAPE = (DIM_X,DIM_Y,3)

    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    val_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    train_batches = train_gen.flow_from_directory(train_dir,target_size=(DIM_X, DIM_Y),batch_size=batch_size, class_mode='binary')
    val_batches = val_gen.flow_from_directory(val_dir, target_size=(DIM_X, DIM_Y) , batch_size=batch_size, class_mode='binary')

    model = create_model((DIM_X,DIM_Y,3))

    early_stopping = EarlyStopping(patience=10)

    checkpointer = ModelCheckpoint('output/resnet50_1024_TL_best.h5', verbose=1, save_best_only=True)
    
    classes = list(iter(train_batches.class_indices))

    for c in train_batches.class_indices:
        print("Class indices: " + str(c))
        classes[train_batches.class_indices[c]] = c
    model.classes = classes

    print("Encoded model classes: " + str(model.classes))

    history = model.fit_generator(train_batches, steps_per_epoch=num_train_steps, epochs=n_epochs, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)

    model.save('resnet50_1024_TL.h5')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument("--datapath", "-d", help="", type=str, default= '../data/')
    parser.add_argument("--batchsize", "-b", help="", type=int, default = 2)
    parser.add_argument("--epochs", "-e", help="", type=int, default = 2)
    parser.add_argument("--cores", "-c", help="", type=str, default = '2')

    args = parser.parse_args()
    print("args: ", args)

    DATA_DIR =args.datapath
    BATCH_SIZE = args.batchsize
    n_epochs = args.epochs

    os.environ['MKL_NUM_THREADS'] = args.cores
    os.environ['GOTO_NUM_THREADS'] = args.cores
    os.environ['OMP_NUM_THREADS'] = args.cores
    os.environ['openmp'] = 'True'

    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')

    create_folder('output')

    main(TRAIN_DIR, VALID_DIR, n_epochs, BATCH_SIZE, DATA_DIR, args.cores)

    