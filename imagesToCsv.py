#!/usr/bin/python

from PIL import Image
import numpy as np 
import os
import sys
import time
import labels
import csv

# The size of the image patch in rows or columns
# Patches are always square
patchsize = 224
channels = 3
yLabel = []

# Set the paths with suffic _patchsize (the image width of the patches)
trainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchsize)
valImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchsize)
testImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchsize)

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

imTrainImgs = np.empty((trainLen,patchsize,patchsize,channels), dtype=float)
imValImgs = np.empty((valLen,patchsize,patchsize,channels), dtype=float)
imTestImgs = np.empty((testLen,patchsize,patchsize,channels), dtype=float)

# Create the csv files (Train, Validation, Test)
X_trainCsv = open(trainImgPath+'/'+'X_train_set_'+str(patchsize), 'w+')
Y_trainCsv = open(trainImgPath+'/'+'Y_train_set_'+str(patchsize), 'w+')

'''
xTrainWriter = csv.writer(X_trainCsv)
yTrainWriter = csv.writer(Y_trainCsv)
'''
X_valCsv = open(valImgPath+'/'+'X_val_set_'+str(patchsize), 'w+')
Y_valCsv = open(valImgPath+'/'+'Y_val_set_'+str(patchsize), 'w+')
'''
xValWriter = csv.writer(X_valCsv)
yValWriter = csv.writer(Y_valCsv)
'''
X_testCsv = open(testImgPath+'/'+'X_test_set_'+str(patchsize), 'w+')
Y_testCsv = open(testImgPath+'/'+'Y_test_set_'+str(patchsize), 'w+')
'''
xTestWriter = csv.writer(X_testCsv)
yTestWriter = csv.writer(Y_testCsv)
'''
start_time = time.time()
# Extract Raw images and save in csv row-wise
# Each row contains an image width*height*channels
print('Train set...')
index = 0
for label in sorted(os.listdir(trainImgPath)):
	if label not in labels.listLabels:
		continue
	for image in sorted(os.listdir(trainImgPath+'/'+label)):
		im = Image.open(trainImgPath+'/'+label+'/'+image)
		imTrainImgs[index] = np.array(im)
		yLabel.append(labels.labels[label])
		index += 1
		
np.savez_compressed(X_trainCsv, imTrainImgs)
np.savez_compressed(Y_trainCsv, np.array(yLabel))
imTrainImgs = None
X_trainCsv.close()
Y_trainCsv.close()
del yLabel[:]

print('Validation Set...')
index = 0
for label in sorted(os.listdir(valImgPath)):
	if label not in labels.listLabels:
		continue
	for image in sorted(os.listdir(valImgPath+'/'+label)):
		im = Image.open(valImgPath+'/'+label+'/'+image)
		imValImgs[index] = np.array(im)
		yLabel.append(labels.labels[label])
		index += 1

np.savez_compressed(X_valCsv, imValImgs)
np.savez_compressed(Y_valCsv, np.array(yLabel))
imValImgs = None
X_valCsv.close()
Y_valCsv.close()
del yLabel[:]

print('Test set...')
index = 0
for label in sorted(os.listdir(testImgPath)):
	if label not in labels.listLabels:
		continue
	for image in sorted(os.listdir(testImgPath+'/'+label)):
		im = Image.open(testImgPath+'/'+label+'/'+image)
		imTestImgs[index] = np.array(im)
		yLabel.append(labels.labels[label])

np.savez_compressed(X_testCsv, imTestImgs)
np.savez_compressed(Y_testCsv, np.array(yLabel))
imTestImgs = None
X_testCsv.close()
Y_testCsv.close()

print("--- %s seconds ---" % (time.time() - start_time))