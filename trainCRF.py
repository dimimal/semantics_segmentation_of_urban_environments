#!/usr/bin/python
# This script is for CRF Training
# 
from __future__ import print_function
import numpy as np
import os
import sys
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
import labels
from cv2 import imread, imwrite
from keras_contrib.layers import CRF 
from dataGenerator import DataGenerator
from keras.models import Model, Sequential, Input
from keras.callbacks import ModelCheckpoint, CSVLogger
from skimage.io import imread, imsave
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
#%matplotlib inline

labels = {
    0 :(128, 64,128),
    1 : (244, 35,232),
    2 : ( 70, 70, 70),
    3 : (102,102,156),
    4 : (190,153,153),
    5 : (153,153,153),
    6 : (250,170, 30),
    7 : (220,220,  0),
    8 : (107,142, 35),
    9 : (152,251,152),
    10 :( 70,130,180),
    11 : (220, 20, 60),
    12 : (255,  0,  0),
    13 : (  0,  0,142),
    14 : (  0,  0, 70),
    15 : (  0, 60,100),
    16 : (  0, 80,100),
    17 : (  0,  0,230),
    18 : (119, 11, 32),
    19 : (255, 255, 255)}

#path = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/Training_Predictions/'
trainFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/Training_Predictions/'
valFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/Validation_Predictions/'
testFolder = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/Test_Predictions/'
imagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_512/X_train_set_512_0001.npz'

patchSize = 512
(img_rows, img_cols) = patchSize, patchSize 
NUM_CLASSES = 20
BATCH_SIZE = 4
modelIndex = 1

checkpointPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/checkpoint_Weights_CRF_{}_{}.h5'.format(modelIndex, patchSize)
lrCurvesPath = '/home/dimitris/GitProjects/semantics_segmentation_of_urban_environments/csv_curves_CRF_{}_{}.h5'.format(modelIndex, patchSize)

def loadCRF(x):
    d = dcrf.DenseCRF2D(x.shape[1], x.shape[2], 20)
    x = np.reshape(x, (-1,x.shape[1]*x.shape[2]))
    U = unary_from_softmax(x)
    print(U.shape)
    print(U[:,0])
    U = -np.log(U)
    print(U[:,0])
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    return d

def loadfiles():
    files = []
    for file in os.listdir(path):
        files.append(file)
    return files

def loadImage(files):
    for file in files:
        x = np.load(path+file)
        return x

def label2color(file):
    array = np.empty((file.shape[0], file.shape[1], 3), dtype='uint8')
    for index, value in np.ndenumerate(file):  
        #print(index)
        array[index[0], index[1]]= np.asarray(labels[value])
    return array

def kerasCRF():
    # Instantiate data generator object
    data_gen = DataGenerator(NUM_CLASSES, BATCH_SIZE, img_rows, img_cols, trainFolder, valFolder, testFolder, normalize=False)
    # Logger callback for learning curves
    csv_logger = CSVLogger(lrCurvesPath, append=True, separator=',')

    # Checkpoint to save the weights with the best validation accuracy.
    checkPoint = ModelCheckpoint(checkpointPath,
                monitor='val_loss',
                verbose=1,
                save_best_only=True,
                save_weights_only=True,
                mode='min')


    model = Sequential()
    crf = CRF(512*NUM_CLASSES, sparse_target=True, input_shape=(img_rows, img_cols, NUM_CLASSES))
    #model.add(Input(shape=(img_rows,img_cols,NUM_CLASSES)))
    model.add(crf)
    model.summary()
    model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
    history = model.fit_generator(generator=data_gen.nextTrain(),
                                      steps_per_epoch=data_gen.getSize(mode='Train')//batch_size,
                                      epochs=epochs,
                                      verbose=1, 
                                      validation_data=data_gen.nextVal(),
                                      validation_steps=data_gen.getSize(mode='Val')//batch_size,
                                      callbacks=[checkPoint, csv_logger])

    data_gen.computeTestClasses()
    print("--- %s seconds ---" % (time.time() - start_time))
    save_model_params(model)
    y_pred = model.predict_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size, verbose=1)
    score = model.evaluate_generator(data_gen.nextTest(), data_gen.getSize(mode='Test')//batch_size)
    print(score)

def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics, reports per classs recall, precision and F1 score'''
    y_true = numpy.asarray(y_true).ravel()
    y_pred = numpy.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('', 'recall', 'precision', 'f1-score', 'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report])
    N = len(y_true)
    print(formatter('avg / total', sum(report2[0]) / N, sum(report2[1]) / N, sum(report2[2]) / N, N) + '\n')

"""
Function which returns the labelled image after applying CRF

"""

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_image = The final output image after applying CRF
#Use_2d = boolean variable 
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied
def crf(original_image, annotated_image, output_image, use_2d = True):
    
    # Converting annotated image to RGB if it is Gray scale
    '''        
    if(len(annotated_image.shape)<3):
        annotated_image = gray2rgb(annotated_image)
    
    imsave("testing2.png",annotated_image)

    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:,:,0] + (annotated_image[:,:,1]<<8) + (annotated_image[:,:,2]<<16)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    print("No of labels in the Image are ")
    print(n_labels)
    '''
    n_labels = 19 # hack?
    #print(annotated_image.shape)
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF(original_image.shape[1]*original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        annotated_image = np.reshape(annotated_image, (-1, original_image.shape[0]*original_image.shape[1]))
        U = unary_from_softmax(-np.log(annotated_image))
        U = np.ascontiguousarray(U)
        #U = unary_from_softmax(sm)
        d.setUnaryEnergy(U)
        print(original_image.shape[:2])
        #print(annotated_image.shape)
        feats = create_pairwise_gaussian(sdims=(20, 20), shape=original_image.shape[:2])
        # This adds the color-independent term, features are the locations only.
        '''
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        '''
        #print(feats.shape)
        d.addPairwiseEnergy(feats, compat=3,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features --
        # because the segmentation that we get from CNN are too coarse
        # and we can use local color features to refine them
        
        feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                           img=original_image, chdim=2)

        d.addPairwiseEnergy(feats, compat=10,
                             kernel=dcrf.DIAG_KERNEL,
                             normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(10)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))
    #print(MAP.shape)
    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    #MAP = colorize[MAP,:]   
    imsave(output_image, MAP)
    return MAP

def testCRF():
    #files = loadfiles()
    original_image = np.load(imagePath)
    original_image = original_image[0]
    
    annotated_image = np.load(trainFolder+'X_train_Predictions_1.npy')
    annotated_image = annotated_image[0,:,:,:-1]
    
    #annotated_image = np.load('image_predictions.txt')
    #annotated_image = np.reshape(annotated_image, (512,512,20))
    output1 = crf(original_image, annotated_image, 'output_image_crf.png')
    output1 = label2color(output1)

    annotated_image = np.argmax(annotated_image, axis=-1)
    annotated_image = label2color(annotated_image)
    plt.figure()
    plt.imshow(annotated_image)
    plt.figure()
    plt.imshow(original_image)
    plt.figure()
    plt.imshow(output1)
    plt.show()

def main():
    #kerasCRF()
    '''
    files = loadfiles()
    x = loadImage(files)
    d = loadCRF(x)
    Q = d.inference(20)
    MAP = np.argmax(Q, axis=0)
    MAP = np.reshape(MAP, (x.shape[1], x.shape[2]))
    MAP = label2color(MAP)
    #MAP = np.asarray(MAP, dtype='uint8')
    imwrite('crf_image_test.png', MAP)
    '''
    testCRF()

if __name__ == '__main__':
    main()