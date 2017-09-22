#!/usr/bin/python

import sys
import os
from skimage import io
import numpy as np
import labels
import re
import time

# How many images you want to cut into patches
# set to None to extract all of them
imageSet = 100

#global patchSize

patchSize = 35
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################

# imagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train/aachen'
# Train set Paths
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train'
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set'

# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/val'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set'

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set'

######################################################
# gtFine Paths
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train'
outTrainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train_set'

valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/val'
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
def imagePatchExtractor(image, file, city, imagepath, finepath, outpath):
	counter = 0
	index = 0
	h, w, c  = image.shape
	i = 0
	j= 0
	
	labelImage = io.imread(finepath+'/'+city+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern)

	while i < h :
		while j < w:
			# Check boundaries
			if j > w or j > w-patchSize:
				croppedImage = image[i:i+patchSize,w-patchSize::,:]
				label = labelImage[i:i+patchSize,w-patchSize::]
			else:
				croppedImage = image[i:i+patchSize,j:j+patchSize,:]
				label = labelImage[i:i+patchSize,j:j+patchSize]
			
			centerLabel = label[patchSize/2, patchSize/2]
			if centerLabel == 255:
				j += patchSize
				counter += 1
				continue

			try:
				rename = file.replace(rawImagePattern, '{0:07d}'.format(counter) +'_'+rawImagePattern)
				io.imsave(outpath+'/'+ labels.listLabels[centerLabel]+'/'+rename, croppedImage)
			except Exception as e:
				print centerLabel
				raise e
				sys.exit(-1)

			j += patchSize
			counter += 1
		
		i += patchSize
		# Leave the last row pixels (max patchsize pixels abandoned)
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
		if not os.path.isdir(outTrainImgPath+'/'+classFolder):
			os.mkdir(outTrainImgPath+'/'+classFolder)
		if not os.path.isdir(outValImgPath+'/'+classFolder):
			os.mkdir(outValImgPath+'/'+classFolder)
		if not os.path.isdir(outTestImgPath+'/'+classFolder):
			os.mkdir(outTestImgPath+'/'+classFolder)
		
		'''
		if classFolder not in os.listdir(outTrainImgPath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outValFinePath):
			os.mkdir(classFolder)
		if classFolder not in os.listdir(outTestFinePath):
			os.mkdir(classFolder)
		'''

##########################################
# Main function 
##########################################
def main():
	

	# Check if the folders for each class are in place
	folderCheck()
	
	# Extract Raw images
	counter = 0
	for city in sorted(os.listdir(trainImagePath)):
		for file in sorted(os.listdir(trainImagePath+'/'+city)):
			if imageSet is not None and counter == imageSet:
				break
			image = io.imread(trainImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, city, trainImagePath, trainFinePath, outTrainImgPath)
			counter += 1

	counter = 0
	for city in sorted(os.listdir(valImagePath)):
		for file in sorted(os.listdir(valImagePath+'/'+city)):
			if imageSet is not None and counter == imageSet:
				break
			image = io.imread(valImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, city, valImagePath, valFinePath, outValImgPath)
			counter += 1

	counter = 0
	for city in sorted(os.listdir(testImagePath)):
		for file in sorted(os.listdir(testImagePath+'/'+city)):
			if imageSet is not None and counter == imageSet:
				break
			image = io.imread(testImagePath+'/'+city+'/'+file)
			#image = image/255.0
			imagePatchExtractor(image, file, city, testImagePath, testFinePath, outTestImgPath)
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
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))