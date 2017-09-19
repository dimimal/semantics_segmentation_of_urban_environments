#!/usr/bin/python

import sys
import os
import re
from skimage import io
import numpy as np

# How many images you want to cut into patches
imageSet = 100

global patchSize

patchSize = 35
rawImagePattern = 'leftImg8bit.png'
fineAnnotPattern = 'labelTrainIds.png'

# Add the file paths
imagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train/aachen'
fineAnnotPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train/aachen'
outTrainPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_patches'
outAnnotPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train_annot_patches'

#############################################
# Extracts the patches from the image and 
# saves to path
#############################################
def imagePatchExtractor(image, file, path):
	counter = 0
	index = 0
	print file
	print image.shape
	h, w, c  = image.shape
	i = 0
	j= 0

	while i < h :
		while j < w:
			if j > w or j > w-patchSize:
				croppedImage = image[i:i+patchSize,w-patchSize::,:]
			else:
				croppedImage = image[i:i+patchSize,j:j+patchSize,:]

			if re.search(rawImagePattern, file, flags=0):
				rename = file.replace(rawImagePattern, str(counter)+'_'+rawImagePattern)
				io.imsave(outTrainPath+'/'+rename, croppedImage)
			elif re.search(fineAnnotPattern, file, flags=0):
				rename = file.replace(fineAnnotPattern, str(counter)+'_'+fineAnnotPattern)
				io.imsave(outAnnotPath+'/'+rename, croppedImage)
			
			j += patchSize
			counter += 1
		
		i += patchSize
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

			if re.search(rawImagePattern, file, flags=0):
				rename = file.replace(rawImagePattern, str(counter)+'_'+rawImagePattern)
				io.imsave(outTrainPath+'/'+rename, croppedImage)
			elif re.search(fineAnnotPattern, file, flags=0):
				rename = file.replace(fineAnnotPattern, str(counter)+'_'+fineAnnotPattern)
				io.imsave(outAnnotPath+'/'+rename, croppedImage)
			
			j += patchSize
			counter += 1
		
		i += patchSize
		if i > h-patchSize:
			return
		j = 0


##########################################
# Main function 
##########################################
def main():
	# Extract Raw images
	'''
	counter = 0
	for file in sorted(os.listdir(imagePath)):
		print file
		if counter == imageSet:
			break
		image = io.imread(imagePath+'/'+file)
		#image = image/255.0
		imagePatchExtractor(image, file, imagePath)
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
		trainLabelsPatchExtractor(image,file, fineAnnotPath)
		counter += 1

if __name__ == '__main__':
	main()