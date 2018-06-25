#!/usr/bin/python

'''
This script loads a keras model from json and 
outputs the predictions from images 
'''

from __future__ import print_function
from lib.utils import computeGlobalIou, plot_confusion_matrix, load_model, medianFilter, getFileLists, load_image, alpha_coefficients
from sklearn.metrics import f1_score, classification_report, confusion_matrix, jaccard_similarity_score, accuracy_score
from lib.bilinearUpsampling import BilinearUpSampling2D
from lib.buildModels import CRFRNN, bilinear_CNN, SD_CNN 
from keras.optimizers  import Adam
from lib.weighted_categorical_crossentropy import weighted_loss
import numpy as np
import os
import labels 
from keras.preprocessing import image 
from keras.models import Model, Sequential
import sys
import re
import argparse 

NUM_CLASSES         = 20
BATCH_SIZE          = 1
PATCHSIZE           = 512
CHANNELS            = 3
IMG_ROWS, IMG_COLS  = PATCHSIZE, PATCHSIZE

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-w', '--weights', default=None, const=None, help='The absolute path of the weights of the model', type=str)
    parser.add_argument('-m', '--model',  default=None, const=None, help='The absolute path of the model in json format', type=str)
    parser.add_argument('-if', '--image_folder', default=None, const=None, help='The absolute \
        path of the folder containing the images for evaluation', type=str)
    parser.add_argument('-gf', '--gt_folder',  default=None, const=None, help='The absolute \
        path of the folder containing the grounf truth images for evaluation', type=str)
    parser.add_argument('-im', '--image', default=None, const=None, help='The absolute path of a single image', type=str)
    parser.add_argument('-mf', '--medianFilter', default=None, const=None, help='The size of the median filter kernel', type=int)

    return parser.parse_args()

def evaluate_model(model, imageList, gtList, kernelSize=None):
    #imageList, gtList = getFileLists(imagepath, gtpath)
    predictions   = np.array([])
    cfAllmatrices = np.empty((NUM_CLASSES,NUM_CLASSES),np.float32)
    metrics       = np.array([])
    globalMetrics = np.array([])
    print(len(gtList), len(imageList))
    assert len(imageList) == len(gtList) 

    for imagePath in imageList:
        # Load image and G.T image
        colorImg  = load_image(imagePath, PATCHSIZE, show=False, scale=True)
        gtImage   = image.load_img(gtList[imageList.index(imagePath)])
        gtImage   = image.img_to_array(gtImage)
        gtImage   = gtImage[:,:,0]
        #
        y_pred = model.predict(colorImg, verbose=1)
        y_pred = np.argmax(y_pred, axis=-1)
        y_pred = np.asarray(y_pred.squeeze(), dtype='uint8')
        colorImg  = resizeImage(y_pred) # Resize to full scale
        # Create sample weight vector to exclude the evaluation of void label
        sampleWeights = np.where(gtImage==255, 0, 1) 

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
    # Write the image in place
    cv.imwrite(os.path.join(os.getcwd(), 'image_predictions_{}'.format(re.findall('\w+_*.png', imagePath)[0])), image)

def main(args):
    # Image prediction or evaluation path
    if not args.image:
        mode = 'eval'
    else:
        mode = 'image'

    if args.model:
        model = load_model(args.model)
    else: 
        raise Exception('Model not given {}'.format(args.model))
    
    # Get the coefficients compile the model
    coefficients = alpha_coefficients()
    # Compile the model
    model.compile(loss=weighted_loss(NUM_CLASSES, coefficients),
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])
    
    # Load Weights
    if args.weights is not None:    
        if os.path.exists(args.weights):
            model.load_weights(args.weights)
            print('Weights loaded successfully')       

    if mode == 'eval':
        if os.path.exists(args.image_folder) and os.path.exists(args.gt_folder):
            imagePaths, gtPaths = getFileLists(args.image_folder, args.gt_folder)
            evaluate_model(model, imagePaths, gtPaths, kernelSize=args.medianFilter)
        else:
            raise Exception('Paths not found {} : {}'.format(args.image_folder, args.gt_folder))
    elif mode == 'image':
        if os.path.exists(args.image):
            predict_image(model, imagepath, kernelSize=args.medianFilter)
        else: 
            raise Exception('Path not found {}'.format(args.image))
            sys.exit(-1)

if __name__ == '__main__':
    args = argument_parser()
    main(args)