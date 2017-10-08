#!/usr/bin/python

from __future__ import print_function

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1,l2
from keras import backend as K
from testCallback import TestCallback 

trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set'
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set'
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set'

trainSetSize = 100
valSetSize = 100
testSetSize = 100 

#trainSetSize = 10000
#valSetSize = 2000
#testSetSize = 1000 
#weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/weights_1_50.h5'

np.random.seed(25)

batch_size = 32
num_classes = 19   
epochs = 3
img_rows, img_cols = 35, 35
input_shape=(img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5),
                activation='selu',
                input_shape=input_shape,
                kernel_initializer='glorot_uniform'))

#BatchNormalization(axis=-1)
model.add(Conv2D(128, 
                padding='valid',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))


#BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Flatten())

#BatchNormalization()
#model.add(Dense(128, activation='selu', kernel_initializer='glorot_uniform'))
#BatchNormalization()
#model.add(Dense(1024, activation='selu', kernel_initializer='glorot_uniform'))
#BatchNormalization()
model.add(Dense(1024, 
            activation='selu', 
            #kernel_regularizer=l1(0.001),
            #activity_regularizer=l2(0.001), 
            kernel_initializer='glorot_uniform'))

model.add(Dropout(0.5, seed=25))
#BatchNormalization()
model.add(Dense(1024, 
            activation='selu',
            #kernel_regularizer=l1(0.001),
            #activity_regularizer=l2(0.001),  
            kernel_initializer='glorot_uniform'))

model.add(Dropout(0.5, seed=25))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#model.load_weights(weightsPath)
# Data Generation
train_datagen = ImageDataGenerator(
                rescale=1./255,
                horizontal_flip=True,
                width_shift_range=0.08, 
                shear_range=0.3,
                height_shift_range=0.08,
                zoom_range=0.08)       

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Tensorboard callback settings
tbCallBack = keras.callbacks.TensorBoard(
            log_dir='./Graph', 
            histogram_freq=0, 
            write_graph=True, 
            write_images=True)

# Initialize the Generators
trainGenerator = train_datagen.flow_from_directory(
            trainFolder,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

validationGenerator = val_datagen.flow_from_directory(
            valFolder,
            target_size=(img_rows, img_cols),
            batch_size=batch_size,
            class_mode='categorical')

testGenerator = test_datagen.flow_from_directory(
            testFolder,
            target_size=(img_rows,img_cols),
            batch_size=batch_size,
            class_mode='categorical')

testCb = TestCallback(epochs, testGenerator, batch_size, testSetSize)


history = model.fit_generator(
            trainGenerator,
            steps_per_epoch=trainSetSize//batch_size,
            epochs=epochs,
            verbose=1, 
            validation_data=validationGenerator,
            validation_steps=valSetSize//batch_size,
            #callbacks=[tbCallBack, TestCallback(epochs, testGenerator, batch_size, testSetSize)])
            callbacks=[tbCallBack, testCb])

'''
score = model.evaluate_generator(
            testGenerator, 
            steps=testSetSize//batch_size, 
            use_multiprocessing=True)
'''
# Print the test error and accuracy of the model
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])
#print(model.TestCallback.getItem(callback))
#print(testCb.score)
# Learning Curves Plots
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(testCb.score[:][1])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation','test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(testCb.score[:][0])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation','test'], loc='upper left')
plt.show()

model_json = model.to_json()
plot_model(model, to_file='model.png')
with open('model_22.json', 'w') as jsonFile:
    jsonFile.write(model_json) 
model.save_weights('weights_model_24.h5')
