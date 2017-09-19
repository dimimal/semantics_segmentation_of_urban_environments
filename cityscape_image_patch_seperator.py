#!/usr/bin/python

import sys
import os
import re
from skimage import io
import numpy as np

# How many images you want to cut into patches
imageSet = 100
#counter = 0
global patchSize

patchSize = 35
rawImagePattern = 'leftImg8bit.png'
fineAnnotPattern = 'gtFine_color.png'

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
	h, w, c = image.shape
	#size = h * w
	i = 0
	j= 0
	#print image.shape
	while i < h :
		while j < w:
			if j > w or j > w-patchSize:
				croppedImage = image[i:i+patchSize,w-patchSize,:]
			else:
				croppedImage = image[i:i+patchSize,j:j+patchSize,:]

			if re.search(rawImagePattern, file, flags=0):
				rename = file.replace(rawImagePattern, str(counter)+'_'+rawImagePattern)
				io.imsave(outTrainPath+'/'+rename, croppedImage)
			elif re.search(fineAnnotPattern, file, flags=0):
				rename = file.replace(fineAnnotPattern, str(counter)+'_'+fineAnnotPattern)
				io.imsave(fineAnnotPath+'/'+rename, croppedImage)
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
	counter = 0
	for file in sorted(os.listdir(imagePath)):
		print file
		if counter == imageSet:
			break
		image = io.imread(imagePath+'/'+file)
		image = image/255.0
		imagePatchExtractor(image, file, imagePath)
		counter += 1

	counter = 0
	for file in sorted(os.listdir(fineAnnotPath)):
		if counter == imageSet:
			break
		image = io.imread(fineAnnotPath+'/'+file)
		image = image/255.0
		imagePatchExtractor(image,file, fineAnnotPath)
		counter += 1

if __name__ == '__main__':
	main()