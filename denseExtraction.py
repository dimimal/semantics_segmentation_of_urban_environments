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
import cv2

# How many images you want to cut into patches
# set to None to extract all of them
trainImageSet       = None
valImageSet         = None
testImageSet        = None
offset              = 200 # how many samples per file
patchSize           = 640
img_rows, img_cols  = 640, 960

rawImagePattern = 'leftImg8bit.png'
finePattern     = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_{}/'.format(patchSize)
outValImgPath   = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_{}/'.format(patchSize)
outTestImgPath  = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_{}/'.format(patchSize)

# Train set Path
trainImagePath  = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train'
# Validation set Path
valImagePath    = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/val'
# Test set Path
testImagePath   = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test'

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
def denseExtractor(imageSet, imagepath, finepath, outpath, filePattern):
    counter = 0
    index   = 0
    skip    = 0      # skip the first # images

    skipIndex = 0 #Keep index of the skipped images
    fileIndex = 1

    imArray = np.array([])
    yLabels = np.array([])

    for subdir, dirs, files in os.walk(imagepath):
        for file in files:
            if skipIndex < skip:
                skipIndex +=1
                continue

            image = io.imread(os.path.join(subdir, file))
            h, w, c  = image.shape

            labelImage = io.imread(os.path.join(finepath, re.findall('\w+_\d+_', file)[0]+ finePattern))

            image = cv2.resize(image, (img_cols, img_rows), cv2.INTER_CUBIC)[:, :, ::-1]
            labelImage = cv2.resize(labelImage, (img_cols, img_rows), cv2.INTER_NEAREST)

            image = np.expand_dims(image, axis=0)
            labelImage = np.expand_dims(labelImage, axis=0)

            im = np.array(image)
            imLabels = np.array(labelImage)
            imLabels = np.clip(imLabels, 0, 19)

            # 2nd Try
            if imArray.ndim == 1:
                imArray = im
                yLabels = imLabels
            else:
                imArray = np.concatenate((imArray, im), axis=0)
                yLabels = np.concatenate((yLabels, imLabels), axis=0)
            # counter += 1

            if index == offset-1:
                x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
                y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')

                np.save(x_Handler, imArray)
                np.save(y_Handler, yLabels)

                fileIndex += 1
                x_Handler.close()
                y_Handler.close()

                # Reset the arrays for refill
                imArray = np.array([])
                yLabels = np.array([])
                print(index)
                index = 0
                print(outpath+filePattern[0])
                print('{}  Saved...'.format(fileIndex))
                continue
            index   += 1

    # Check if the file handlers are closed with the residual samples
    if imArray.size > 0:
        x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
        y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')

        np.save(x_Handler, imArray)
        np.save(y_Handler, yLabels)
        x_Handler.close()
        y_Handler.close()
        imArray = np.array([])
        yLabels = np.array([])


def main():
    print('Train...')
    filePattern = ['X_train_set_', 'Y_train_set_']
    denseExtractor(trainImageSet, trainImagePath, trainFinePath, outTrainImgPath, filePattern)
    print('Validation...')
    filePattern = ['X_validation_set_', 'Y_validation_set_']
    denseExtractor(valImageSet, valImagePath, valFinePath, outValImgPath, filePattern)
    print('Test...')
    filePattern = ['X_test_set_', 'Y_test_set_']
    denseExtractor(testImageSet, testImagePath, testFinePath, outTestImgPath, filePattern)


if __name__ == '__main__':
    start_time = time.time()
    main()
    print('----- %s  seconds -----'%(time.time()-start_time))