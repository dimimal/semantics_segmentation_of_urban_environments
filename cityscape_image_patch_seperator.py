#!/usr/bin/python

import sys
import os
from skimage import io
import numpy as np
import labels

# How many images you want to cut into patches
imageSet = 1

global patchSize

patchSize = 35
rawImagePattern = 'leftImg8bit.png'
fineAnnotPattern = 'labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################

# imagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train/aachen'
# Train set Paths
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train'
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set'

# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set'

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set'

######################################################
# gtFine Paths
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train'
outTrainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train_set'

valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/validation'
outValFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/validation_set'

testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/test'
outTestFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/test_set'

# labelTrainIds 
#fineAnnotPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train/aachen'
#outAnnotPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train_annot_patches'

#############################################
# Extracts the patches from the image and 
# saves to path
#############################################
def imagePatchExtractor(image, file, imagepath, finepath):
	counter = 0
	index = 0
	print file
	print image.shape
	h, w, c  = image.shape
	i = 0
	j= 0

	while i < h :
		while j < w:
			# Check boundaries
			if j > w or j > w-patchSize:
				croppedImage = image[i:i+patchSize,w-patchSize::,:]
			else:
				croppedImage = image[i:i+patchSize,j:j+patchSize,:]

			rename = file.replace(rawImagePattern, str(counter)+'_'+rawImagePattern)
			io.imsave(outTrainPath+'/'+rename, croppedImage)
						
			j += patchSize
			counter += 1
		
		i += patchSize
		# Leave the last row pixels (max 35 pixels abandoned)
		if i > h-patchSize:
			return
		j = 0


######################################################
# Extracts the patches from TrainLabelIds images
# Pixel contains the id of the class
######################################################
def trainLabelsPatchExtractor(image, file, path):
	counter = 0
	index = 0
	print file
	print image.shape
	h, w = image.shape
	i = 0
	j= 0

	while i < h :
		while j < w:
			# Check bounds
			if j > w or j > w-patchSize:
				croppedImage = image[i:i+patchSize,w-patchSize::]
			else:
				croppedImage = image[i:i+patchSize,j:j+patchSize]

			centerLabel = croppedImage[patchSize/2,patchSize/2]
			
			rename = file.replace(fineAnnotPattern, str(counter)+'_'+'L_'+ '{0:03d}'.format(centerLabel) +'_'+fineAnnotPattern )
			io.imsave(outAnnotPath+'/'+rename, croppedImage)
			
			j += patchSize
			counter += 1
		
		i += patchSize
		if i > h-patchSize:
			return
		j = 0

############################################################
# Checks if the folders for our extracted patches of images
# are in place, each folder(train,val,test) has 19 folders
# one of each class
###########################################################
def folderCheck():
	for classFolder in labels.listLabels:
		if classFolder not in os.listdir(outTrainPath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outValPath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outTestPath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outTrainFinePath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outValFinePath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outTestFinePath):
			os.mkdir(classFolder)


##########################################
# Main function 
##########################################
def main():
	

	# Check if the folders for each class are in place
	folderCheck()
	

	
	# Extract Raw images
	counter = 0
	for city in sorted(os.listdir(trainImagePath)):
		for file in city:
			if counter == imageSet:
				break
			image = io.imread(trainImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, trainImagePath, trainFinePath)
			counter += 1
	
	counter = 0
	for city in sorted(os.listdir(valImagePath)):
		for file in city:
			if counter == imageSet:
				break
			image = io.imread(valImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, valImagePath, valFinePath)
			counter += 1

	counter = 0
	for city in sorted(os.listdir(testImagePath)):
		for file in city:
			if counter == imageSet:
				break
			image = io.imread(testImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, testImagePath, testFinePath)
			counter += 1

	'''
	# Extract Label images
	counter = 0
	for file in sorted(os.listdir(fineAnnotPath)):
		if counter == imageSet:
			break
		if fineAnnotPattern not in file:
			continue
		image = io.imread(fineAnnotPath+'/'+file)
		#image = image/255.0 
		trainLabelsPatchExtractor(image, file, fineAnnotPath)
		counter += 1
	'''
if __name__ == '__main__':
	main()