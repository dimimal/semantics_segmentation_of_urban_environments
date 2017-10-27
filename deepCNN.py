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
from sklearn.metrics import confusion_matrix, f1_score
#from testCallback import TestCallback 
import labels
import time


patchSize = 70
trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

#trainSetSize = 467827
#valSetSize = 224391
#testSetSize = 77657 

trainSetSize = 100
valSetSize = 200
testSetSize = 3000 
#weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/weights_1_50.h5'

np.random.seed(25)

batch_size = 32
num_classes = 19   
channels = 3
epochs = 3
img_rows, img_cols = 70, 70
input_shape=(img_rows, img_cols, channels)

start_time = time.time()

model = Sequential()
'''
model.add(Conv2D(32, kernel_size=(3,3),
                strides=(2,2),
                padding='same',
                activation='selu',
                input_shape=input_shape,
                kernel_initializer='glorot_uniform'))

BatchNormalization(axis=-1)
model.add(Conv2D(32, 
                strides=(2,2),
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))


model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
'''
model.add(Conv2D(64, kernel_size=(3,3),
                strides=(2,2),
                padding='same',
                input_shape=input_shape,
                activation='selu',
                kernel_initializer='glorot_uniform'))

model.add(Conv2D(64, 
                padding='same',
                strides=(2,2),                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)

model.add(Conv2D(128, kernel_size=(3,3),
                padding='same',
                activation='selu',
                kernel_initializer='glorot_uniform'))

BatchNormalization(axis=-1)
model.add(Conv2D(128, 
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))
BatchNormalization(axis=-1)
model.add(Conv2D(256, 
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
BatchNormalization(axis=-1)
model.add(Conv2D(256, 
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
BatchNormalization(axis=-1)
model.add(Conv2D(512, 
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
BatchNormalization(axis=-1)
model.add(Conv2D(512, 
                padding='same',                
                kernel_size=(3,3), 
                activation='selu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
model.add(Flatten())

#BatchNormalization()
#model.add(Dense(128, activation='selu', kernel_initializer='glorot_uniform'))
#BatchNormalization()
#model.add(Dense(1024, activation='selu', kernel_initializer='glorot_uniform'))
BatchNormalization()
model.add(Dense(2048, 
            activation='selu', 
            kernel_regularizer=l2(0.01),
            activity_regularizer=l2(0.01), 
            kernel_initializer='glorot_uniform'))
BatchNormalization()
model.add(Dropout(0.3))
model.add(Dense(1024, 
            activation='selu', 
            kernel_regularizer=l2(0.01),
            activity_regularizer=l2(0.01), 
            kernel_initializer='glorot_uniform'))

BatchNormalization()
model.add(Dropout(0.3))
model.add(Dense(512, 
            activation='selu',
            kernel_regularizer=l2(0.01),
            activity_regularizer=l2(0.01),  
            kernel_initializer='glorot_uniform'))
BatchNormalization()
model.add(Dropout(0.3))
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

# Instantiate callback object for testing on every epoch
#testCb = TestCallback(epochs, testGenerator, batch_size, testSetSize)


history = model.fit_generator(
            trainGenerator,
            steps_per_epoch=trainSetSize//batch_size,
            epochs=epochs,
            verbose=1, 
            validation_data=validationGenerator,
            validation_steps=valSetSize//batch_size,
            callbacks=[tbCallBack])

print("--- %s seconds ---" % (time.time() - start_time))

# Get the predictions on the test set and calculate the F1-score
y_pred = model.predict_generator(testGenerator, steps=testSetSize//batch_size, verbose=1)
f1Score = f1_score(testGenerator.classes[:y_pred.shape[0]], np.argmax(y_pred, axis=1), average='micro')
print('F1 score:: {}'.format(f1Score))
#cfMatrix = confusion_matrix(testGenerator.classes[:y_pred.shape[0]], np.argmax(y_pred, axis=1), labels=labels.idLabelsList)

# Learning Curves Plots
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

model_json = model.to_json()
plot_model(model, to_file='model.png')
with open('model_26.json', 'w') as jsonFile:
    jsonFile.write(model_json) 
model.save_weights('weights_model_26.h5')
