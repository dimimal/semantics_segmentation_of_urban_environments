#!/usr/bin/python

'''
imageDownScale script downsamples an input image as long with the fine annotated image
from 2048x1024 to 512x256 pixels
'''
from __future__ import print_function
import time
import os
import re
import matplotlib.pyplot as plt
import cv2 as cv
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################

# Train set Paths
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train'
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_train'

# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/val'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_validation'

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/refined_Test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_test'

######################################################
# Configure paths for gtFine labeled image set
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train'
outTrainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_train'

valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/val'
outValFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_validation'

testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/refined_Test'
outTestFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_test'

'''
Function:
downSample:
			input: 
				image: The rgb image 
				annotImage: The fine annotated image
				file: The name of the image(file)
				outPath: The path which outputs the image
				outAnnot: The path of the resized annotated image
'''	
def downSample(image, annotImage, file, outPath, outAnnot):
	resizedImage = cv.resize(image, dsize=(image.shape[0]/2,image.shape[0]/4))
	resizedAnnotImage = cv.resize(annotImage, dsize=(annotImage.shape[0]/2,annotImage.shape[0]/4), interpolation=cv.INTER_NEAREST)
	cv.imwrite(outPath+'/'+file, resizedImage)
	cv.imwrite(outAnnot+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern, resizedAnnotImage)

'''
Main Function
'''
def main():
	'''
	print('Train extraction....')
	counter = 0
	for city in sorted(os.listdir(trainImagePath)):
		for file in sorted(os.listdir(trainImagePath+'/'+city)):
			#if trainImageSet is not None and counter == trainImageSet:
			#	break
			image = cv.imread(trainImagePath+'/'+city+'/'+file)
			labelImage = cv.imread(trainFinePath+'/'+city+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern, cv.IMREAD_GRAYSCALE)
			downSample(image, labelImage, file, outTrainImgPath, outTrainFinePath)
			print('Train:: ', counter)
			counter += 1
	
	print('Validation extraction....')
	counter = 0
	for city in sorted(os.listdir(valImagePath)):
		for file in sorted(os.listdir(valImagePath+'/'+city)):
			#if valImageSet is not None and counter == valImageSet:
			#	break
			image = cv.imread(valImagePath+'/'+city+'/'+file)
			labelImage = cv.imread(valFinePath+'/'+city+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern, cv.IMREAD_GRAYSCALE)
			downSample(image, labelImage, file, outValImgPath, outValFinePath)
			print('Validation:: ', counter)
			counter += 1
	'''
	print('Testing extraction....')
	counter = 0
	for city in sorted(os.listdir(testImagePath)):
		for file in sorted(os.listdir(testImagePath+'/'+city)):
			#if testImageSet is not None and counter == testImageSet:
			#	break
			image = cv.imread(testImagePath+'/'+city+'/'+file)
			labelImage = cv.imread(testFinePath+'/'+city+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern, cv.IMREAD_GRAYSCALE)
			downSample(image, labelImage, file, outTestImgPath, outTestFinePath)
			#image = image/255.0
			print('Test:: ', counter)
			counter += 1

if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))