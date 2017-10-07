#!/usr/bin/python
# This script extracts patches from cityscapes dataset images 
# leftImg8bit and gtFine image set
	
import sys
import os
from skimage import io
import numpy as np
import labels
import re
import time

# How many images you want to cut into patches
# set to None to extract all of them
imageSet = 150

patchSize = 35
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################

# Train set Paths
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train'
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set'

# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/val'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set'

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/refined_Test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set'

######################################################
# Configure paths for gtFine labeled image set
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train'
valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/val'
testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/refined_Test'


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


##########################################
# Main function 
##########################################
def main():
	

	# Check if the folders for each class are in place
	folderCheck()
	
	# Extract Raw images
	'''
	counter = 0
	for city in sorted(os.listdir(trainImagePath)):
		for file in sorted(os.listdir(trainImagePath+'/'+city)):
			if imageSet is not None and counter == imageSet:
				break
			image = io.imread(trainImagePath+'/'+city+'/'+file)
			imagePatchExtractor(image, file, city, trainImagePath, trainFinePath, outTrainImgPath)
			counter += 1
	'''
	counter = 0
	for city in sorted(os.listdir(valImagePath)):
		for file in sorted(os.listdir(valImagePath+'/'+city)):
			if imageSet is not None and counter == imageSet:
				break
			image = io.imread(valImagePath+'/'+city+'/'+file)
			imagePatchExtractor(image, file, city, valImagePath, valFinePath, outValImgPath)
			counter += 1
	'''
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
if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))