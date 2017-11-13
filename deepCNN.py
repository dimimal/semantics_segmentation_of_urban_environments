#!/usr/bin/python

from __future__ import print_function

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.utils import plot_model
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, AlphaDropout, Activation, LocallyConnected2D
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1,l2
from keras import backend as K
from keras.constraints import MaxNorm
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from mlxtend.plotting import plot_confusion_matrix
from weighted_categorical_crossentropy import weighted_categorical_crossentropy
import labels
import time
import cv2 as cv

patchSize = 16
channels = 3

class_weights = {0:0.366039785069,
1:0.0583057048131,
2:0.243340573911,
3:0.00554475820281,
4:0.00548759574711,
5:0.0111466788613,
6:0.00225791700011,
7:0.0062307076712,
8:0.170772836401,
9:0.0169772493426,
10:0.0285812278495,
11:0.00820281239282,
12:0.000400137189894,
13:0.0693094775352,
14:0.000771693151938,
15:0.00254372927861,
16:0.000943180519035,
17:0.000857436835486,
18:0.00228649822796
}

weights = [key for id,key in class_weights.iteritems()]

trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

trainSetSize = 150000
valSetSize = 30000
testSetSize = 100912 
'''
trainSetSize = 200
valSetSize = 100
testSetSize = 100 
'''

np.random.seed(25)

batch_size = 32
num_classes = 19   

channels = 3
epochs = 20
img_rows, img_cols = patchSize, patchSize
input_shape=(img_rows, img_cols, channels)

def preprocess(image):
    return cv.cvtColor(image, cv.COLOR_RGB2HSV)

start_time = time.time()

model = Sequential()
model = Sequential()
model.add(Conv2D(8, kernel_size=(3, 3),
                #kernel_constraint=MaxNorm(4.),
                #kernel_constraint=max_norm(3),
                #activation='selu',
                input_shape=input_shape,
                kernel_initializer='lecun_uniform'))

BatchNormalization(axis=-1)
model.add(Activation('selu'))

model.add(Conv2D(16, kernel_size=(3,3),
                #kernel_constraint=MaxNorm(4.),
                padding='valid',
                #activation='selu',
                kernel_initializer='lecun_uniform'))
BatchNormalization(axis=-1)
model.add(Activation('selu'))

model.add(Conv2D(32, kernel_size=(3,3),
                #kernel_constraint=MaxNorm(4.),
                padding='valid',
                #activation='selu',
                kernel_initializer='lecun_uniform'))
BatchNormalization(axis=-1)
model.add(Activation('selu'))
'''
model.add(Conv2D(128, kernel_size=(3,3),
                #kernel_constraint=MaxNorm(4.),
                padding='valid',
                #activation='selu',
                kernel_initializer='lecun_uniform'))
BatchNormalization(axis=-1)
model.add(Activation('selu'))
'''
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(AlphaDropout(0.2))
BatchNormalization()
'''
model.add(Dense(64, 
            #activation='selu',
            #kernel_regularizer=l2(0.0001),
            #activity_regularizer=l2(0.0001),  
            kernel_initializer='lecun_uniform'))
BatchNormalization()

model.add(Activation('selu'))
'''
model.add(Flatten())
model.add(Dense(128, activation='selu', kernel_initializer='lecun_uniform'))
#model.add(AlphaDropout(0.2))
model.add(Dense(num_classes, activation='softmax', kernel_initializer='lecun_uniform'))

model.summary()
model.compile(loss=weighted_categorical_crossentropy(weights),
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# Data Generators Instatiate
train_datagen = ImageDataGenerator(
                rescale=1./255,
                #featurewise_center=True,
                #featurewise_std_normalization=True,
                #zca_whitening=False
                fill_mode='reflect', 
                horizontal_flip=True,
                #zoom_range=0.2,
                preprocessing_function=preprocess)       

val_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess)
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=preprocess)

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

# Early stopping to avoid overfitting.
earlyStopping = EarlyStopping(monitor='val_loss', patience=7) 

# Checkpoint to save the weights with the best validation accuracy.
#bestModelPath = stamp + '.h5'
checkPoint = ModelCheckpoint('checkPointWeights.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto')

plateauCallback = ReduceLROnPlateau(monitor='loss',
            factor=0.5,
            patience=4,
            min_lr=0.00005,
            verbose=1,
            cooldown=2)
#earlyStopping = EarlyStopping(monitor='val_loss', patience=2) 

history = model.fit_generator(
            trainGenerator,
            steps_per_epoch=trainSetSize//batch_size,
            epochs=epochs,
            verbose=1,
            class_weight=None, 
            validation_data=validationGenerator,
            validation_steps=valSetSize//batch_size,
            callbacks=[checkPoint, plateauCallback, earlyStopping])

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
plot_model(model, to_file='model_16_raw.png')

with open('model_16_raw.json', 'w') as jsonFile:
    jsonFile.write(model_json) 
model.save_weights('weights_model_16_raw.h5')
