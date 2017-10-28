#!/usr/bin/python

from PIL import Image
import numpy as np 
import os
import sys
import time
import labels

# The size of the image patch in rows or columns
# Patches are always square
patchsize = 224
channels = 3
yLabel = []
offset = 1000 # Defines the chunk size sampling from the dataset 

# Set the paths with suffic _patchsize (the image width of the patches)
trainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchsize)
valImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchsize)
testImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchsize)

# Find the total number of images in the dataset (train, val, test)
# to initialize the np arrays
'''
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

start_time = time.time()


# Extract Raw images and save in csv row-wise
# Each row contains an image width*height*channels
print('Train set...')
fileIndex = 1

x_trainHandler = open(trainImgPath+'/'+'X_train_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
y_trainHandler = open(trainImgPath+'/'+'Y_train_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')

index = 0
for label in os.listdir(trainImgPath):
	if label not in labels.listLabels:
		continue
	for image in os.listdir(trainImgPath+'/'+label):
		im = np.array(Image.open(trainImgPath+'/'+label+'/'+image))
		np.save(x_trainHandler, im)
		np.save(y_trainHandler, labels.labels[label])
		if index == offset-1:
			fileIndex += 1
			x_trainHandler.close()
			y_trainHandler.close()

			x_trainHandler = open(trainImgPath+'/'+'X_train_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			y_trainHandler = open(trainImgPath+'/'+'Y_train_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			index = 0
			continue
		index += 1

# sanity check for the remaining samples to be saved		
if not x_trainHandler.closed and not y_trainHandler.closed:
	x_trainHandler.close()
	y_trainHandler.close() 


print('Validation Set...')
fileIndex = 1

x_valHandler = open(valImgPath+'/'+'X_val_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
y_valHandler = open(valImgPath+'/'+'Y_val_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')

index = 0
for label in os.listdir(valImgPath):
	if label not in labels.listLabels:
		continue
	for image in os.listdir(valImgPath+'/'+label):
		im = np.array(Image.open(valImgPath+'/'+label+'/'+image))
		np.save(x_valHandler, im)
		np.save(y_valHandler, labels.labels[label])
		if index == offset-1:
			fileIndex += 1
			x_valHandler.close()
			y_valHandler.close()

			x_valHandler = open(valImgPath+'/'+'X_val_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			y_valHandler = open(valImgPath+'/'+'Y_val_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			index = 0
			continue
		index += 1

# sanity check for the remaining samples to be saved		
if not x_valHandler.closed and not y_valHandler.closed:
	x_trainHandler.close()
	y_trainHandler.close() 

print('Test set...')
fileIndex = 1

x_testHandler = open(testImgPath+'/'+'X_test_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
y_testHandler = open(testImgPath+'/'+'Y_test_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')

index = 0
for label in os.listdir(testImgPath):
	if label not in labels.listLabels:
		continue
	for image in os.listdir(testImgPath+'/'+label):
		im = np.array(Image.open(testImgPath+'/'+label+'/'+image))
		np.save(x_testHandler, im)
		np.save(y_testHandler, labels.labels[label])
		if index == offset-1:
			fileIndex += 1
			x_testHandler.close()
			y_testHandler.close()

			x_testHandler = open(testImgPath+'/'+'X_test_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			y_testHandler = open(testImgPath+'/'+'Y_test_set_'+str(patchsize)+'_'+'%04d.npy'%(fileIndex), 'a+')
			index = 0
			continue
		index += 1

# sanity check for the remaining samples to be saved		
if not x_testHandler.closed and not y_testHandler.closed:
	x_testHandler.close()
	y_testHandler.close() 

print("--- %s seconds ---" % (time.time() - start_time))