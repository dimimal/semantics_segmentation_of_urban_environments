#!/usr/bin/python

'''
This script loads a keras model from json and 
outputs the predictions from images 
'''
from __future__ import print_function
#from keras.preprocessing import image
#import keras
#import matplotlib.pyplot as plt
#from keras import backend as K
#from lib.dataGenerator import DataGenerator 
from lib.utils import computeGlobalIou, plot_confusion_matrix, load_model, medianFilter, getFileLists
from sklearn.metrics import f1_score, classification_report, confusion_matrix, jaccard_similarity_score, accuracy_score
#from lib.weighted_categorical_crossentropy import weighted_loss
#from lib.bilinearUpsampling import BilinearUpSampling2D
from lib.buildModels import CRFRNN, bilinear_CNN, SD_CNN 
import numpy as np
import os
import labels  
from keras.models import Model, Sequential
import sys
#import time
#import sys
import re
import argparse 

NUM_CLASSES         = 20
BATCH_SIZE          = 1
PATCHSIZE           = 512
CHANNELS            = 3
IMG_ROWS, IMG_COLS  = PATCHSIZE, PATCHSIZE

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-w', '--weights', nargs='?', default=None, const=None, help='The absolute path of the weights of the model', type=str)
    parser.add_argument('-m', '--model', nargs='?', default=None, const=None, help='The absolute path of the model in json format', type=str)
    parser.add_argument('-file', '--file', nargs='?', default=None, const=None, help='The absolute \
        path of the folder containing the images for evaluation', type=str)
    parser.add_argument('-im', '--image', nargs='?', default=None, const=None, help='The absolute path of a single image', type=str)
    parser.add_argument('-mf', '--medianFilter', nargs='?', default=None, const=None, help='The size of the median filter kernel', type=int)

    return parser.parse_args()

def evaluateModel(model, kernelSize=None):
    imageList, gtList = getFileLists(imagepath, gtpath)
    predictions   = np.array([])
    
    cfAllmatrices = np.empty((NUM_CLASSES,NUM_CLASSES),np.float32)
    metrics       = np.array([])
    globalMetrics = np.array([])

    for imagePath in imageList:
        # Load image and G.T image
        colorImg  = load_image(imagePath, patchSize, show=False, scale=True)
        gtImage   = keras.preprocessing.image.load_img(gtList[imageList.index(imagePath)])
        gtImage   = keras.preprocessing.image.img_to_array(gtImage)
        gtImage   = gtImage[:,:,0]
        #
        y_pred = model.predict(colorImg, verbose=1)
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = np.asarray(y_pred.squeeze(), dtype='uint8')
        colorImg  = resizeImage(y_pred) # Resize to full scale
        # Create sample weight vector to exclude 
        sampleWeights = np.where(gtImage==255, 0, 1) # Mask the gt image to set void label to zero

        # Enable median Filter 
        if kernelSize:
            colorImg = medianFilter(colorImg, ksize=kernelSize)
        #
        cfMatrix     = confusion_matrix(gtImage.flatten(), colorImg.flatten(), labels=listLabels, sample_weight=sampleWeights.flatten())
        globalIoU    = computeGlobalIou(cfMatrix)
        #
        metrics       = np.append(metrics, sampleIoU)
        globalMetrics = np.append(globalMetrics, globalIoU)
        print("Global IoU:{}".format(globalIoU))
    
    print('Average global mean IoU: {}'.format(np.average(globalMetrics)))

    return np.average(globalMetrics) 

def predict_image(model, imagePath, kernelSize=None, show=False):
    predictions = np.array([])
    image = load_image(imagePath, patchSize=512, show=False, scale=True)
    for index in np.ndindex(image.shape[0]):
        patch = np.expand_dims(image[index], axis=0)
        y_pred = model.predict(patch, verbose=1)
    image = np.argmax(y_pred, axis=-1)
    image = np.asarray( image.squeeze(), dtype='uint8')
    # Resize to full scale
    image = resizeImage(image) 
    if kernelSize:
        image = medianFilter(image, ksize=19)
    image = labels2Image(image)
    cv.imwrite(os.path.join(os.getcwd(), 'image_predictions_{}'.format(re.findall('\w+_*.png', imagePath)[0])), image)

def main(args):
    # Image prediction or evaluation path
    if not args.im:
        mode = 'eval'
    else:
        mode = 'image'

    if args.model:
        model = load_model(args.model, weights=args.weights)

    else: 
        raise Exception('Model not given {}'.format(args.model))
    
    if mode == 'eval':
        imagePath, gtPath = getFileLists(realTestPath , realTestGtPath)
        evaluate_model(model, imagePath, gtPath, kernelSize=args.medianFilter)
    elif mode == 'image':
        if os.path.exists(args.image):
            predict_image(model, imagepath, kernelSize=args.medianFilter)
        else: 
            raise Exception('Path not found {}'.format(args.image))
            sys.exit(-1)

if __name__ == '__main__':
    args = argument_parser()
    main(args)