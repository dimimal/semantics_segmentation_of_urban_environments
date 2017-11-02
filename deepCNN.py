#!/usr/bin/python

from __future__ import print_function

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, AlphaDropout 
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1,l2
from keras import backend as K
from keras.constraints import max_norm
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
import labels
import time


patchSize = 140
channels = 3

#patchSize = 70

trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

#trainSetSize = 467827
#valSetSize = 224391
#testSetSize = 77657 
'''
trainSetSize = 29868
valSetSize = 14725
testSetSize = 4571 
'''

trainSetSize = 200
valSetSize = 100
testSetSize = 100 

np.random.seed(25)

batch_size = 32
num_classes = 19   

channels = 3
epochs = 3
img_rows, img_cols = patchSize, patchSize
input_shape=(img_rows, img_cols, channels)

start_time = time.time()

model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5),
                #kernel_constraint=max_norm(3),
                activation='selu',
                input_shape=input_shape,
                kernel_initializer='lecun_uniform'))

model.add(Conv2D(16, kernel_size=(5, 5),
                #kernel_constraint=max_norm(3),
                activation='selu',
                #input_shape=input_shape,
                kernel_initializer='lecun_uniform'))
#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)
'''
model.add(Conv2D(32, 
                padding='valid',                
                kernel_size=(3,3), 
                activation='elu',
                kernel_initializer='lecun_uniform'))
'''

#BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(3,3)))
model.add(Conv2D(32, kernel_size=(3, 3),
                #kernel_constraint=max_norm(3),
                activation='selu',
                #input_shape=input_shape,
                kernel_initializer='lecun_uniform'))
model.add(Conv2D(32, kernel_size=(3, 3),
                #kernel_constraint=max_norm(3),
                activation='selu',
                #input_shape=input_shape,
                kernel_initializer='lecun_uniform'))

model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(3,3),
                #kernel_constraint=max_norm(3),
                activation='selu',
                kernel_initializer='lecun_uniform'))

model.add(Conv2D(64, kernel_size=(3,3),
                #kernel_constraint=max_norm(3),
                activation='selu',
                kernel_initializer='lecun_uniform'))

#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)

model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.1))

#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)
#model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#BatchNormalization()
#model.add(Dense(128, activation='elu', kernel_initializer='lecun_uniform'))
#BatchNormalization()
#model.add(Dense(1024, activation='elu', kernel_initializer='lecun_uniform'))
#BatchNormalization()
model.add(Dense(1024, 
            activation='selu', 
            #kernel_regularizer=l2(0.0001),
            #activity_regularizer=l2(0.0001), 
            kernel_initializer='lecun_uniform'))
#model.add(AlphaDropout(0.3))
#BatchNormalization()
model.add(Dense(512, 
            activation='selu',
            #kernel_regularizer=l2(0.0001),
            #activity_regularizer=l2(0.0001),  
            kernel_initializer='lecun_uniform'))
#model.add(AlphaDropout(0.3))

model.add(Dense(num_classes, activation='softmax', kernel_initializer='lecun_uniform'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
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

#earlyStopping = EarlyStopping(monitor='val_loss', patience=2) 

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
print(classification_report(testGenerator.classes[:y_pred.shape[0]], np.argmax(y_pred, axis=1)))
cfMatrix = confusion_matrix(testGenerator.classes[:y_pred.shape[0]], np.argmax(y_pred, axis=1))
fig, ax = plot_confusion_matrix(conf_mat=cfMatrix)
plt.show()
f1Score = f1_score(testGenerator.classes[:y_pred.shape[0]], np.argmax(y_pred, axis=1), average='macro')
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
