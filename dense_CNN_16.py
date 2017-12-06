#!/usr/bin/python

from __future__ import print_function
import os
import labels
import numpy as np
import time
import keras
import matplotlib.pyplot as plt
import random
from keras.utils import plot_model
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, AlphaDropout, Activation, Reshape, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Lambda, core, Add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2
from keras.constraints import MaxNorm
from keras import backend as K
from keras.activations import softmax
from weighted_categorical_crossentropy import weighted_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from dataGenerator import DataGenerator
from sklearn.metrics import classification_report, confusion_matrix, jaccard_similarity_score
from BillinearUpsampling import BilinearUpSampling2D
from keras_contrib.layers import CRF
import itertools

debug_mode = False
patchSize = 32
modelIndex = 2
channels = 3
# =============================== Declare the dataset Paths ===========================================================================
trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_{}'.format(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_{}'.format(patchSize)
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_{}'.format(patchSize)

# ============================ Declare the paths for the parameters of our network ====================================================
weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/Weights_dense_{}_{}.h5'.format(modelIndex, patchSize)
checkpointPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/checkpoint_Weights_dense_{}_{}.h5'.format(modelIndex, patchSize)
lrCurvesPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/dense_lrCurves_{}_{}.csv'.format(modelIndex, patchSize)
modelParamPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/model_params_{}_{}.json'.format(modelIndex, patchSize)
modelPicturePath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/model_dense_{}_{}.png'.format(modelIndex, patchSize)
reportPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/report_dense_{}_{}.txt'.format(modelIndex, patchSize)

np.random.seed(25)

batch_size = 16
num_classes = 20   
epochs = 5
img_rows, img_cols = patchSize, patchSize
input_shape=(img_rows, img_cols, channels)

# Define the weights for each class, zero for unlabeled class
class_weights ={0:1.,
                1:1.,
                2:1.,
                3:1.,
                4:1.,
                5:1.,
                6:1.,
                7:1.,
                8:1.,
                9:1.,
                10:1.,
                11:1.,
                12:1.,
                13:1.,
                14:1.,
                15:1.,
                16:1.,
                17:1.,
                18:1.,
                19:0.0}

weights = [key for index,key in class_weights.iteritems()]

start_time = time.time()

# Calculate the mean IoU 
def computeMeanIou(matrix):
    # Exclude tha last column and row, omitting void label
    # from score
    matrix = np.delete(matrix, matrix.shape[0]-1, axis=0)
    matrix = np.delete(matrix, matrix.shape[1]-1, axis=1)
    tp = np.diag(matrix)
    fp = matrix - np.diag(np.diag(matrix))
    fn = matrix - np.diag(np.diag(matrix))

    fp = np.sum(fp, axis=0)
    fn = np.sum(fn, axis=1)
    perClassIou = np.true_divide(tp, (tp+fp+fn))
    meanIoU = np.true_divide(np.sum(perClassIou), num_classes)
    return perClassIou, meanIoU 

# Keras model function for 16x16 patch size    
def fcn_16(patchSize, channels):
    input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)
    # Block 1
    x = AtrousConvolution2D(16, (3,3), atrous_rate=(2,2) ,padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_1_16')(inputs)
    x = Conv2D(16, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_2_16')(x)
    x = Conv2D(16, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_3_16')(x)
    # Block 2
    x = AtrousConvolution2D(32, (3,3), atrous_rate=(3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_4_32')(x)
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_5_32')(x)
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_6_32')(x)
    # Block 3
    x = AtrousConvolution2D(64, (3,3), atrous_rate=(4,4), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_7_64')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_8_64')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_3_64')(x)

    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_9_128')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_10_128')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_11_128')(x)
    x = AlphaDropout(0.1)(x)
    #x = MaxPooling2D(pool_size=(2, 2), name='pool_1')(x)
    # Block 4
    #x = Deconvolution2D(128, (5,5), activation='selu', kernel_initializer='lecun_normal', name='Deconv_1_128')(x)
    #x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_12_128')(x)
    #x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_13_128')(x)
    # Block 5 (Deconv Block)
    #x = Deconvolution2D(64, (3,3), activation='selu', kernel_initializer='lecun_normal', name='Deconv_2_64')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_14_64')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_15_64')(x)
    x = AlphaDropout(0.1)(x)
    #x = Deconvolution2D(32, (3,3), activation='selu', kernel_initializer='lecun_normal', name='Deconv_3_32')(x)
    #x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_16_32')(x)
    #x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_17_32')(x)
    #
    x = Conv2D(num_classes, (1,1), kernel_initializer='lecun_normal', name='out_conv')(x)

    predictions = core.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(loss=weighted_categorical_crossentropy(weights) ,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    '''
    if weightsPath:
        model.load_weights(weightsPath)
    '''
    return model
    
def fcn_32(patchSize, channels):
    input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(inputs)
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    conv_1 = Conv2D(128, (3,3), dilation_rate=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    conv_1 = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(conv_1)
    conv_1 = Conv2D(64, (1,1), padding='valid', activation='selu', kernel_initializer='lecun_normal')(conv_1)

    conv_2 = Conv2D(128, (3,3), dilation_rate=(3, 3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    conv_2 = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(conv_2)
    conv_2 = Conv2D(64, (1,1), padding='valid', activation='selu', kernel_initializer='lecun_normal')(conv_2)
    #
    conv_3 = Conv2D(128, (3,3), dilation_rate=(5, 5), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    conv_3 = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(conv_3)
    conv_3 = Conv2D(64, (1,1), padding='valid', activation='selu', kernel_initializer='lecun_normal')(conv_3)
    #
    x = Add()([conv_1, conv_2, conv_3])
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal')(x)

    predictions = core.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(loss=weighted_categorical_crossentropy(weights) ,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    
    if checkpointPath:
        model.load_weights(checkpointPath)
    
    return model

def fcn_128(patchSize, channels):
    input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (5,5), activation='selu', padding='same', kernel_initializer='lecun_normal')(inputs)
    x = Conv2D(64, (5,5), activation='selu', padding='same', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #    
    x = UpSampling2D()(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = UpSampling2D()(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = UpSampling2D()(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = AlphaDropout(0.1)(x)
    #
    x = Conv2D(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal')(x)

    predictions = core.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(loss=weighted_categorical_crossentropy(weights) ,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    '''
    if weightsPath:
        model.load_weights(weightsPath)
    
    '''
    return model

def fcn_64(patchSize, channels):
    input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (5,5), dilation_rate=(3, 3), activation='selu', padding='same', kernel_initializer='lecun_normal')(inputs)
    x = Conv2D(64, (5,5), dilation_rate=(3, 3), activation='selu', padding='same', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #    
    x = UpSampling2D()(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = UpSampling2D()(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = UpSampling2D()(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = AlphaDropout(0.1)(x)
    #
    x = Conv2D(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal')(x)

    predictions = core.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(loss=weighted_categorical_crossentropy(weights) ,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    
    if os.path.exists(weightsPath):
        model.load_weights(weightsPath)
    
    
    return model

def fcn_512(patchSize, channels):
    input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)
    # Block 1
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(inputs)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    #
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal')(x)
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal')(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2,2))(x)
    # Atrous 1 block
    atrous_1 = Conv2D(256, (3,3), dilation_rate=(3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    atrous_1 = BilinearUpSampling2D(size=(2, 2))(atrous_1)
    atrous_1 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal')(atrous_1)
    #
    atrous_2 = Conv2D(256, (3,3), dilation_rate=(6,6), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    atrous_2 = BilinearUpSampling2D(size=(2, 2))(atrous_2)
    atrous_2 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal')(atrous_2)
    #   
    atrous_3 = Conv2D(256, (3,3), dilation_rate=(9,9), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    atrous_3 = BilinearUpSampling2D(size=(2, 2))(atrous_3)
    atrous_3 = Conv2D(128, (1,1), padding='same',  activation='selu', kernel_initializer='lecun_normal')(atrous_3)
    #
    atrous_4 = Conv2D(256, (3,3), dilation_rate=(12,12), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    atrous_4 = BilinearUpSampling2D(size=(2, 2))(atrous_4)
    atrous_4 = Conv2D(128, (1,1), padding='same',  activation='selu', kernel_initializer='lecun_normal')(atrous_4)
    #atrous_4 = Conv2DTranspose(128, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal')(atrous_4)
    x = Add()([atrous_1, atrous_2, atrous_3, atrous_4])
    x = BilinearUpSampling2D(size=(2, 2))(x)    
    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    x = BilinearUpSampling2D(size=(2, 2))(x)
    
    x = Conv2DTranspose(num_classes, (1,1), padding='valid')(x)
    
    #y = core.Reshape((patchSize,patchSize,num_classes))(x)
    predictions = core.Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.summary()
    model.compile(loss=weighted_categorical_crossentropy(weights),
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    '''
    if weightsPath:
        model.load_weights(weightsPath)
    
    '''
    return model

def plot_results(history, y_true, y_pred, score):    
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = np.argmax(y_pred, axis=-1)
    y_true = np.argmax(y_true, axis=-1)
    y_pred = np.reshape(y_pred, (y_pred.shape[0]*img_rows*img_cols))
    y_true = np.reshape(y_true, (y_true.shape[0]*img_rows*img_cols))

    # Learning Curve Plots
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
    report = classification_report(y_true, y_pred)
    print(report)
    meanIoU = jaccard_similarity_score(y_true, y_pred, normalize=True)  

    # Compute mean IoU
    cfMatrix = confusion_matrix(y_true, y_pred)
    labelIou, meanIoUfromcf = computeMeanIou(cfMatrix)
    
    print('----- Mean IoU ----- ')
    print('------ %s -----------'%(str(meanIoU)))
    print('---- Manual mean Iou from CF ------')
    print('------ %s -----------'%(str(meanIoUfromcf)))

    # Remove the last label
    cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=0)
    cfMatrix = np.delete(cfMatrix, cfMatrix.shape[0]-1, axis=1)   
    plt.figure()
    fig, ax = plot_confusion_matrix(cfMatrix, labels.listLabels, normalize=True)
    plt.show()

    # Write report to txt
    with open(reportPath,'w') as fileObj:
        fileObj.write(report)
        fileObj.write(str(meanIoU))
        fileObj.write(str(meanIoUfromcf))

def plot_confusion_matrix(cm, 
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def save_model_params(model):
    model_json = model.to_json()
    plot_model(model, to_file=modelPicturePath)
    with open(modelParamPath, 'w') as jsonFile:
        jsonFile.write(model_json) 
    model.save_weights(weightsPath)


def main():
    # Instantiate callback object for testing on every epoch
    # testCb = TestCallback(epochs, testGenerator, trainGenerator, batch_size, testSetSize)

    # Early stopping to avoid overfitting.
    earlyStopping = EarlyStopping(monitor='val_loss', patience=18) 

    # Logger callback for learning curves
    csv_logger = CSVLogger(lrCurvesPath, append=True, separator=',')

    # Checkpoint to save the weights with the best validation accuracy.
    checkPoint = ModelCheckpoint(checkpointPath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='auto')

    plateauCallback = ReduceLROnPlateau(monitor='loss',
                factor=0.5,
                patience=7,
                min_lr=0.00005,
                verbose=1,
                cooldown=2)

    # Instantiate data generator object
    data_gen = DataGenerator(num_classes, batch_size, img_rows, img_cols, trainFolder, valFolder, testFolder)

    if patchSize == 16:
        model = fcn_16(patchSize, channels)
    elif patchSize == 32:
        #model = fcn_32(patchSize, channels)
        json_file = open(modelParamPath, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json, custom_objects={'BilinearUpSampling2D':BilinearUpSampling2D})
        model.load_weights(checkpointPath)
        model.summary()
        model.compile(loss=weighted_categorical_crossentropy(weights) ,
                  optimizer=keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    elif patchSize == 64:
        model = fcn_64(patchSize, channels)
    elif patchSize == 128:
        model = fcn_128(patchSize, channels)
    elif patchSize == 512:
        model = fcn_512(patchSize, channels)

    # ================================== Start Training ====================================== #
    start_time = time.time()
    if debug_mode:
        history = model.fit_generator(generator=data_gen.nextTrain(),
                                      steps_per_epoch=1,
                                      epochs=1,
                                      verbose=1, 
                                      validation_data=data_gen.nextVal(),
                                      validation_steps=1,
                                      callbacks=[earlyStopping, plateauCallback, checkPoint, csv_logger])
    
        print("--- %s seconds ---" % (time.time() - start_time))
        save_model_params(model)
        y_pred = model.predict_generator(data_gen.nextTest(), steps=1, verbose=1)
        score = model.evaluate_generator(data_gen.nextTest(), steps=1)
    else:
        history = model.fit_generator(generator=data_gen.nextTrain(),
                                      steps_per_epoch=data_gen.getSize(mode='Train')//batch_size,
                                      epochs=epochs,
                                      verbose=1, 
                                      validation_data=data_gen.nextVal(),
                                      validation_steps=data_gen.getSize(mode='Val')//batch_size,
                                      callbacks=[earlyStopping, plateauCallback, checkPoint, csv_logger])

        data_gen.computeTestClasses()
        print("--- %s seconds ---" % (time.time() - start_time))
        save_model_params(model)
        y_pred = model.predict_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size, verbose=1)
        score = model.evaluate_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size)
    y_true = data_gen.getClasses(y_pred.shape[0])
    plot_results(history, y_true, y_pred, score)



if __name__ == '__main__': 
    main()
