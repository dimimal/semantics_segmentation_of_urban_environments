#!/usr/bin/python

from __future__ import print_function

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l1,l2
from keras.constraints import max_norm
from keras import backend as K
from testCallback import TestCallback
from keras.callbacks import EarlyStopping

patchSize = 224
channels = 3
trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

trainSetSize = 30000
valSetSize = 15000
testSetSize = 5000

def checkLength():
# Find the total number of images in the dataset (train, val, test)
# to initialize the np arrays
    trainLen = 0
    for label in sorted(os.listdir(trainImgPath)):
        if label not in labels.listLabels:
            continue
        for image in sorted(os.listdir(trainImgPath+'/'+label)):
            trainLen += 1
    valLen = 0
    for label in sorted(os.listdir(valImgPath)):
        if label not in labels.listLabels:
            continue
        for image in sorted(os.listdir(valImgPath+'/'+label)):
            valLen += 1

    testLen = 0
    for label in sorted(os.listdir(testImgPath)):
        if label not in labels.listLabels:
            continue
        for image in sorted(os.listdir(testImgPath+'/'+label)):
            testLen += 1
'''
# loadData: Loads the dataset from npz files
# Input:
#       path: The path of the file
#       patchSize: The row size of the patches
#       channels: The channels of the array 
#       flag: Set to 'Train' 'Val' or 'Test' for each set 
def loadData(path, patchSize, size, channels, flag):
    if flag  == 'Train':
        X = np.load(path+'/'+'X_train_set_'+str(patchSize)+'.npz', mmap_mode='r')
        Y = np.load(path+'/'+'Y_train_set_'+str(patchSize)+'.npz')
    elif flag == 'Val':
        X = np.load(path+'/'+'X_val_set_'+str(patchSize)+'.npz', mmap_mode='r')
        Y = np.load(path+'/'+'Y_val_set_'+str(patchSize)+'.npz')
    elif flag == 'Test':
        X = np.load(path+'/'+'X_test_set_'+str(patchSize)+'.npz')
        Y = np.load(path+'/'+'Y_test_set_'+str(patchSize)+'.npz')
    print(X.shape)
    #xArray = np.reshape(X,(size, patchSize,patchSize,channels))
    return X, Y

X_train, Y_train = loadData(trainFolder, patchSize, trainSetSize, channels, flag='Train') 
X_val, Y_val = loadData(valFolder, patchSize, valSetSize, channels, flag='Val')
X_test, Y_test = loadData(testFolder, patchSize, testSetSize, channels, flag='Test')
'''
#trainSetSize = 100
#valSetSize = 200
#testSetSize = 100 
#weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/weights_1_50.h5'
def dataGenerator(path, setSize, patchSize):
    index = 0
    offset = 2000
    while index < setSize:
        X = np.memmap(path+'/'+'X_train_set_'+str(patchSize)+'.npy', mmap_mode='r')
        Y = np.load(path+'/'+'Y_train_set_'+str(patchSize)+'.npy')
        if setSize - index < offset:
            yield X[index:setSize], Y[index:setSize]
        else:  
            yield X[index:index+offset], Y[index:index+offset]
        index += offset

np.random.seed(25)

batch_size = 32
num_classes = 19   
epochs = 20
img_rows, img_cols = 224, 224
input_shape=(img_rows, img_cols, channels)

trainLen, valLen, testLen = checkLength()

start_time = time.time()

model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7),
                kernel_constraint=max_norm(3),
                activation='elu',
                input_shape=input_shape,
                kernel_initializer='glorot_uniform'))
#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)
'''
model.add(Conv2D(32, 
                padding='valid',                
                kernel_size=(3,3), 
                activation='elu',
                kernel_initializer='glorot_uniform'))
'''

#BatchNormalization(axis=-1)
model.add(MaxPooling2D(pool_size=(5,5)))
#model.add(Dropout(0.1))
model.add(Conv2D(64, kernel_size=(5,5),
                kernel_constraint=max_norm(3),
                activation='elu',
                kernel_initializer='glorot_uniform'))

model.add(Conv2D(64, kernel_size=(5,5),
                kernel_constraint=max_norm(3),
                activation='elu',
                kernel_initializer='glorot_uniform'))

#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)
model.add(Conv2D(64, 
                padding='valid',                
                kernel_constraint=max_norm(3),
                kernel_size=(3,3), 
                activation='elu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(3,3)))
#model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3,3),
                kernel_constraint=max_norm(3),
                padding='same',
                activation='elu',
                input_shape=input_shape,
                kernel_initializer='glorot_uniform'))
#model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.1))

#BatchNormalization(axis=-1)
model.add(Conv2D(128, 
                padding='same',                
                kernel_constraint=max_norm(3),
                kernel_size=(3,3), 
                activation='elu',
                kernel_initializer='glorot_uniform'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

#BatchNormalization()
#model.add(Dense(128, activation='elu', kernel_initializer='glorot_uniform'))
#BatchNormalization()
#model.add(Dense(1024, activation='elu', kernel_initializer='glorot_uniform'))
BatchNormalization()
model.add(Dense(1024, 
            activation='elu', 
            #kernel_regularizer=l2(0.0001),
            #activity_regularizer=l2(0.0001), 
            kernel_initializer='glorot_uniform'))

BatchNormalization()
model.add(Dense(512, 
            activation='elu',
            #kernel_regularizer=l2(0.0001),
            #activity_regularizer=l2(0.0001),  
            kernel_initializer='glorot_uniform'))
model.add(Dropout(0.5))

model.add(Dense(num_classes, activation='softmax', kernel_initializer='glorot_uniform'))

model.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#model.load_weights(weightsPath)
# Data Generation
train_datagen = ImageDataGenerator(
                featurewise_std_normalization=True,
                featurewise_center=True,
                horizontal_flip=True,
                width_shift_range=0.08, 
                shear_range=0.3,
                height_shift_range=0.08,
                zoom_range=0.08)       

val_datagen = ImageDataGenerator(
            featurewise_std_normalization=True,
            featurewise_center=True)

test_datagen = ImageDataGenerator()

# Tensorboard callback settings
tbCallBack = keras.callbacks.TensorBoard(
            log_dir='./Graph', 
            histogram_freq=0, 
            write_graph=True, 
            write_images=True)

#train_datagen.fit(X_train)

#X_train -= train_datagen.mean()
#X_train /= (train_datagen.std + K.epsilon())

#X_val -= train_datagen.mean()
#X_val /= (train_datagen.std + K.epsilon())

#X_test -= train_datagen.mean()
#X_test /= (train_datagen.std + K.epsilon())

#print(X_test[0:1,1:10,0])
# Initialize the Generators
trainGenerator = train_datagen.flow(
            X_train,
            Y_train,
            shuffle=True,
            batch_size=batch_size)
'''
validationGenerator = val_datagen.flow(
            X_val,
            Y_val,
            shuffle=True,
            batch_size=batch_size)

'''
testGenerator = test_datagen.flow(
            X_test,
            Y_test,
            batch_size=batch_size)

# Instantiate callback object for testing on every epoch
testCb = TestCallback(epochs, testGenerator, trainGenerator, batch_size, testSetSize)
#earlyStopping = EarlyStopping(monitor='val_loss', patience=2) 

for e in range(epochs):
    print("epoch %d" % e)
    for X_train, Y_train in dataGenerator(trainFolder, trainLen, patchSize): 
        for X_batch, Y_batch in train_datagen.flow(X_train, Y_train, shuffle=True, batch_size=batch_size):
            #loss = model.train(X_batch, Y_batch)
            history = model.fit(
                        trainGenerator,
                        steps_per_epoch=trainSetSize//batch_size,
                        epochs=epochs,
                        verbose=1, 
                        validation_data=(X_val, Y_val),
                        validation_steps=valSetSize//batch_size,
                        callbacks=[tbCallBack, testCb])


print("--- %s seconds ---" % (time.time() - start_time))

# Learning Curve Plots
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.plot(testCb.score[:,1])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.plot(testCb.score[:,0])
plt.xlim(0, epochs)
plt.xticks(np.arange(0, epochs+1, 5))
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation','test'], loc='upper left')
plt.show()

model_json = model.to_json()
plot_model(model, to_file='model.png')
with open('model_.json', 'w') as jsonFile:
    jsonFile.write(model_json) 
model.save_weights('weights_model_.h5')
