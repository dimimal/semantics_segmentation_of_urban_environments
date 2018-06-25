#!/usr/bin/env python

"""Change the paths accordingly to extract your npy files from cityscapes
dataset in order to train the models
"""
from __future__ import print_function

import time
import numpy as np
import os
from skimage import io
import re

# How many images you want to cut into patches
# set to None to extract all of them
trainImageSet       = None
valImageSet         = None
testImageSet        = None
offset              = 500 # how many samples per file
patchSize           = 512
img_rows, img_cols  = 512, 512

rawImagePattern = 'leftImg8bit.png'
finePattern     = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_{}/'.format(patchSize)
outValImgPath   = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_{}/'.format(patchSize)
outTestImgPath  = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_{}/'.format(patchSize)

# Train set Path
trainImagePath  = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_train'
# Validation set Path
valImagePath    = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_validation'
# Test set Path
testImagePath   = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_test'

######################################################
# Configure paths for gtFine labeled image set
######################################################
trainFinePath   = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_train'
valFinePath     = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_validation'
testFinePath    = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_test'

#############################################
# Extracts the patches from the image and 
# saves to path(Dense pixel extraction)
#############################################
def denseExtractor(imageSet, imagepath, finepath, outpath, filePattern, mode):
    counter = 0
    index   = 0
    skip    = 0      # skip the first # images
    
    skipIndex = 0 #Keep index of the skipped images
    fileIndex = 1

    x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
    y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
    
    imArray = np.array([])
    yLabels = np.array([])
    
    for file in sorted(os.listdir(imagepath)):
        if skipIndex < skip:
            skipIndex +=1
            continue
        print(counter)
        
        image = io.imread(os.path.join(imagepath, file))
        h, w, c  = image.shape
        # load the annoated image
        labelImage = io.imread(os.path.join(finepath, re.findall('\w+_\d+_\d+_', file)[0]+finePattern))
        im = np.array(image)
        imLabels = np.array(labelImage)
        imLabels = np.clip(imLabels, 0, 19)
       
        # 2nd Try 
        if imArray.size == patchSize*patchSize*c:
            imArray = np.stack((imArray, im), axis=0)
            yLabels = np.append(yLabels, imLabels)
        elif imArray.size == 0:
            imArray = im
            yLabels = np.append(yLabels, imLabels)
        else:
            imArray = np.insert(imArray, index, im, axis=0)
            yLabels = np.append(yLabels, imLabels)
        counter += 1
        if index == offset-1:
            np.save(x_Handler, imArray)
            np.save(y_Handler, yLabels)
            fileIndex += 1
            x_Handler.close()
            y_Handler.close()
            # Reset the arrays for refill
            imArray = np.array([])
            yLabels = np.array([])

            x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
            y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
            index = 0
            print(outpath+filePattern[0])
            print('{}  Saved...'.format(fileIndex))
            continue
        index   += 1

    # Check if the file handlers are closed with the residual samples
    if not x_Handler.closed and not y_Handler.closed:
        np.save(x_Handler, imArray)
        np.save(y_Handler, yLabels)
        x_Handler.close()
        y_Handler.close() 
        imArray = np.array([])
        yLabels = np.array([])

    
def main():
    print('Train...')
    filePattern = ['X_train_set_', 'Y_train_set_']
    denseExtractor(trainImageSet, trainImagePath, trainFinePath, outTrainImgPath, filePattern, mode)
    print('Validation...')
    filePattern = ['X_validation_set_', 'Y_validation_set_']
    denseExtractor(valImageSet, valImagePath, valFinePath, outValImgPath, filePattern, mode)
    print('Test...')
    filePattern = ['X_test_set_', 'Y_test_set_']
    denseExtractor(testImageSet, testImagePath, testFinePath, outTestImgPath, filePattern, mode)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('----- %s  seconds -----'%(time.time()-start_time))