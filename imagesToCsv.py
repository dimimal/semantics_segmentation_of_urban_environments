#!/usr/bin/python

from PIL import Image
import numpy as np 
import os
import sys
import time
import labels
import re

# The size of the image patch in rows or columns
# Patches are always square
patchsize = 224
channels = 3

trainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchsize)
valImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchsize)
testImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchsize)

#im = Image.open('um_000000.png')
X_trainCsv = open('X_train_set_224.csv', 'w+')
Y_trainCsv = open('Y_train_set_224.csv', 'w+')

#re.findall(pattern, trainImgPath)
# Create the header of the csv
header = ''
for num in range(patchsize*patchsize*channels):
 	header += 'pixel_{},'.format(num)
header.rstrip(',') 

counter = 0
# Extract Raw images
for label in sorted(os.listdir(trainImgPath)):
	for image in sorted(os.listdir(trainImgPath+'/'+label)):
		im = Image.open(trainImgPath+'/'+label+'/'+image)
		imArray = np.array(im).flatten()
		#print imArray
		#print imArray.shape
		#print np.max(imArray)
		np.savetxt(X_trainCsv, imArray, fmt='%s', header=header, delimiter=',')
		#X_trainCsv.write(imArray)
		Y_trainCsv.write(str(labels.labels[label])+'\n')
		counter+=1
		if counter == 2:
			break
	break
X_trainCsv.close()
Y_trainCsv.close()