import math, json, os, sys

import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Flatten
import argparse

#DATA_DIR = '../data/'
SIZE = (1024, 1024)
#BATCH_SIZE = 2
#n_epochs = 2 #1000

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set arguments')
    parser.add_argument("--datapath", "-d", help="", type=str, default= '../data/')
    parser.add_argument("--batchsize", "-b", help="", type=int, default = 2)
    parser.add_argument("--epochs", "-e", help="", type=int, default = 2)

    args = parser.parse_args()
    print("args: ", args)

    DATA_DIR =args.datapath
    BATCH_SIZE = args.batchsize
    n_epochs = args.epochs


    TRAIN_DIR = os.path.join(DATA_DIR, 'train')
    VALID_DIR = os.path.join(DATA_DIR, 'valid')



    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples/BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples/BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='binary', shuffle=True, batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='binary', shuffle=True, batch_size=BATCH_SIZE)

    #model = keras.applications.resnet50.ResNet50(include_top=False, input_shape=(1024,1024,3), classes=2) #downloads pretrained Model-->  
    #model.save('resnet50_pretrained_includeTopFalse.h5')
    model = load_model('resnet50_pretrained_includeTopFalse.h5') 
    
    classes = list(iter(batches.class_indices))
    print("used classes: ", classes)
    model.layers.pop()
    for layer in model.layers:
        layer.trainable=False
    last = model.layers[-1].output
    #flatten
    flattened = Flatten()(last)
    #x = Dense(len(classes), activation="softmax")(flattened)
  
    x = Dense(1, activation="sigmoid")(flattened)
    finetuned_model = Model(model.input, x)
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='binary_crossentropy', metrics=['binary_accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=10)
    checkpointer = ModelCheckpoint('resnet50_best.h5', verbose=1, save_best_only=True)

    #print(finetuned_model.summary())
    history = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=n_epochs, callbacks=[early_stopping, checkpointer], validation_data=val_batches, validation_steps=num_valid_steps)
    finetuned_model.save('resnet50_final.h5')

    with open('file.json', 'w') as f:
        json.dump(history.history, f)