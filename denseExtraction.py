#!/usr/bin/env python

from __future__ import print_function
import time
import numpy as np
import os
from skimage import io
import re

# How many images you want to cut into patches
# set to None to extract all of them
trainImageSet = None
valImageSet = None
testImageSet = None
offset = 50000 # how many samples per file
mode = 'Full' # select full for whole image extraction or Patch for patch extraction
patchSize = 32
img_rows, img_cols = 512, 512
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################
if mode == 'Patch':
	outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_'+str(patchSize)+'/'
	outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_'+str(patchSize)+'/'
	outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_'+str(patchSize)+'/'
else:
	outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_train_set_full/'
	outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_validation_set_full/'
	outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/dense_test_set_full/'

# Train set Path
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_train'
# Validation set Path
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_validation'
# Test set Path
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_test'

######################################################
# Configure paths for gtFine labeled image set
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_train'
valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_validation'
testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_test'

#############################################
# Extracts the patches from the image and 
# saves to path(Dense pixel extraction)
#############################################
def denseExtractor(imageSet, imagepath, finepath, outpath, filePattern, mode):
	counter = 0
	index = 0
	skip = 0 	  # skip the first # images
	skipIndex = 0 #Keep index of the skipped images
	fileIndex = 1

	x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
	y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
	
	imArray = np.array([])
	yLabels = np.array([])
	if mode == 'Patch':
		#counter = 0
		#for city in sorted(os.listdir(imagepath)):
		for file in sorted(os.listdir(imagepath)):
			if skipIndex < skip:
				skipIndex +=1
				continue
			print(counter)
			if imageSet is not None and counter == imageSet: # Extact patches from a number of frames
				if not x_Handler.closed and not y_Handler.closed:
					np.save(x_Handler, imArray)
					np.save(y_Handler, yLabels)
					x_Handler.close()
					y_Handler.close() 
					imArray = np.array([])
					yLabels = np.array([])
				return

			image = io.imread(imagepath+'/'+file)
			h, w, c  = image.shape
			# load the annoated image
			labelImage = io.imread(finepath+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			counter += 1
			
			i = 0
			j = 0
			#print '2'
			while i < h :
				while j < w:
					# Check boundaries
					if j > w or j > w-patchSize:
						croppedImage = image[i:i+patchSize,(w-patchSize):,:]
						label = labelImage[i:i+patchSize,(w-patchSize):]
					else:
						croppedImage = image[i:i+patchSize,j:j+patchSize,:]
						label = labelImage[i:i+patchSize,j:j+patchSize]
					#print(label.shape)
					im = np.array(croppedImage)
					imLabels = np.array(label)
					imLabels = np.clip(imLabels, 0, 19)
					#print(imLabels.shape)
					if imArray.size == 0:
						imArray = im
						yLabels = imLabels
					else:
						imArray = np.concatenate((imArray, im))
						yLabels = np.concatenate((yLabels, imLabels))			

					if index == offset-1:
						np.save(x_Handler, imArray)
						np.save(y_Handler, yLabels)
						fileIndex += 1
						x_Handler.close()
						y_Handler.close()
						# Reset the arrays for refill
						imArray = np.array([])
						yLabels = np.array([])

						x_Handler = open(outpath+filePattern[0]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
						y_Handler = open(outpath+filePattern[1]+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
						index = 0
						continue
					index +=1
					j += patchSize
					#counter += 1
				
				i += patchSize
				j = 0
	elif mode == 'Full':	
		for file in sorted(os.listdir(imagepath)):
			if skipIndex < skip:
				skipIndex +=1
				continue
			print(counter)
			if imageSet is not None and counter == imageSet: # Extact patches from a number of frames
				if not x_Handler.closed and not y_Handler.closed:
					np.save(x_Handler, imArray)
					np.save(y_Handler, yLabels)
					x_Handler.close()
					y_Handler.close() 
					imArray = np.array([])
					yLabels = np.array([])
				return

			image = io.imread(imagepath+'/'+file)
			h, w, c  = image.shape
			# load the annoated image
			labelImage = io.imread(finepath+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			im = np.array(image)
			imLabels = np.array(labelImage)
			imLabels = np.clip(imLabels, 0, 19)
			
			if imArray.size == 0:
				imArray = im
				yLabels = imLabels
			else:
				imArray = np.concatenate((imArray, im))
				yLabels = np.concatenate((yLabels, imLabels))
			counter += 1
			
	# Check if the file handlers are closed with the residual samples
	if not x_Handler.closed and not y_Handler.closed:
		np.save(x_Handler, imArray)
		np.save(y_Handler, yLabels)
		x_Handler.close()
		y_Handler.close() 
		imArray = np.array([])
		yLabels = np.array([])
	
def main():
	print('Train...')
	filePattern = ['X_train_set_', 'Y_train_set_']
	denseExtractor(trainImageSet, trainImagePath, trainFinePath, outTrainImgPath, filePattern, mode)
	print('Validation...')
	filePattern = ['X_validation_set_', 'Y_validation_set_']
	#denseExtractor(valImageSet, valImagePath, valFinePath, outValImgPath, filePattern, mode)
	print('Test...')
	filePattern = ['X_test_set_', 'Y_test_set_']
	#denseExtractor(testImageSet, testImagePath, testFinePath, outTestImgPath, filePattern, mode)


if __name__ == '__main__':
	start_time = time.time()
	main()
	print('----- %s  seconds -----'%(time.time()-start_time))