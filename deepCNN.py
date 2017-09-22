#!/usr/bin/python

from __future__ import print_function
import keras
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras import backend as K

trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set'
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set'
weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/weights_1_50.h5'

np.random.seed(25)

batch_size = 32
num_classes = 19   
epochs = 20
img_rows, img_cols = 35, 35
input_shape=(img_rows, img_cols, 3)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                activation='selu',
                input_shape=input_shape,
                kernel_initializer='glorot_normal'))
BatchNormalization(axis=-1)

model.add(Conv2D(64, kernel_size=(3, 3),
                activation='selu', 
                kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))

BatchNormalization(axis=-1)
model.add(Conv2D(64, kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_normal'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

BatchNormalization()
model.add(Dense(128, activation='selu', kernel_initializer='glorot_normal'))
BatchNormalization()

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_normal'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.load_weights(weightsPath)
# Data Generation
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


trainGenerator = train_datagen.flow_from_directory(
            trainFolder,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

validationGenerator = test_datagen.flow_from_directory(
            valFolder,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

model.fit_generator(trainGenerator,
                    steps_per_epoch=2000//batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validationGenerator,
                    validation_steps=800//batch_size)

model.save_weights('weights_1_50.h5')
