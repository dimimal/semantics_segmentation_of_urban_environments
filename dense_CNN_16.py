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
from keras.layers import Input, Dense, Dropout, Flatten, AlphaDropout, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Lambda, core
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1,l2
from keras.constraints import MaxNorm
from keras import backend as K
from keras.activations import softmax
from keras.losses import categorical_crossentropy
from weighted_categorical_crossentropy import weighted_categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from dataGenerator import DataGenerator
from sklearn.metrics import classification_report, confusion_matrix, jaccard_similarity_score
from mlxtend.plotting import plot_confusion_matrix
#from vis.visualization import visualize_cam


debug_mode = True
patchSize = 16
channels = 3

trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_{}'.format(patchSize)
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_{}'.format(patchSize)#+'/NpyFiles'
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_{}'.format(patchSize)#+'/NpyFiles'
weightsPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/Weights_dense_{}.h5'.format(patchSize)
checkpointPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/checkpoint_Weights_dense_{}.h5'.format(patchSize)
lrCurvesPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/dense_lrCurves_1_{}.csv'.format(patchSize)
modelParamPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/model_params_{}.json'.format(patchSize)
modelPicturePath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/model_dense_{}.png'.format(patchSize)
reportPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/report_dense_{}.txt'.format(patchSize)

np.random.seed(25)

batch_size = 16
num_classes = 20   
patchSize = 16
epochs = 20
img_rows, img_cols = patchSize, patchSize
input_shape=(img_rows, img_cols, channels)
# Defiune the weights for each class, zero for unlabeled class
class_weights = {
0:1.,
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

    x = Conv2D(16, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(inputs)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #x = AlphaDropout(0.2)(x)
    #
    x = Conv2D(16, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #x = AlphaDropout(0.2)(x)
    #
    x = Conv2D(32, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #x = AlphaDropout(0.2)(x)
    #
    x = Conv2D(32, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #x = AlphaDropout(0.2)(x)
    #
    x = Conv2D(64, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    #
    x = Conv2D(64, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    x = Conv2D(128, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    x = Conv2D(128, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    x = Conv2D(256, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    x = Conv2D(256, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)
    #
    '''
    x = Conv2D(512, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    #
    x = Conv2D(512, (3,3), padding='same', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    '''
    x = AlphaDropout(0.1)(x)
    #
    x = Conv2D(num_classes, (1,1), padding='valid', kernel_constraint=MaxNorm(3.),   kernel_initializer='lecun_uniform')(x)
    x = BatchNormalization(axis=-1)(x)
    x = Activation('selu')(x)

    y = core.Reshape((patchSize,patchSize,num_classes))(x)
    #y = core.Permute((3,2,1))(y)

    predictions = core.Activation('softmax')(y)

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
    print(classification_report(y_true, y_pred))

    # Write report to txt
    with open(reportPath,'w') as fileObj:
        fileObj.write(classification_report(y_true, y_pred))

    # Compute mean IoU
    meanIoU = jaccard_similarity_score(y_true, y_pred, normalize=True)  

    cfMatrix = confusion_matrix(y_true, y_pred)
    labelIou, meanIoUfromcf = computeMeanIou(cfMatrix)
    fig, ax = plot_confusion_matrix(conf_mat=cfMatrix)
    print('----- Mean IoU ----- ')
    print('------ %s -----------'%(str(meanIoU)))
    print('---- Manual mean Iou from CF ------')
    print('------ %s -----------'%(str(meanIoUfromcf)))
    plt.show()

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

    data_gen = DataGenerator(num_classes, batch_size, img_rows, img_cols, trainFolder, valFolder, testFolder)

    model = fcn_16(patchSize, channels)
    
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

        print("--- %s seconds ---" % (time.time() - start_time))
        y_pred = model.predict_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size, verbose=1)
        score = model.evaluate_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size)
    y_true = data_gen.getClasses(y_pred.shape[0])
    plot_results(history, y_true, y_pred, score)
    save_model_params(model)



if __name__ == '__main__': 
    main()
