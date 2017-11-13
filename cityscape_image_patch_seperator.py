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
trainImageSet = 40
valImageSet = 20
testImageSet = 10
offset = 99999999 # how many samples per file

patchSize = 40
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'

#####################################################
# Configure paths for leftImg8bit image set
#####################################################

# Train set Paths
trainImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_train'
outTrainImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/train_set_'+str(patchSize)

# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_validation'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

######################################################
# Configure paths for gtFine labeled image set
######################################################
trainFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_train'
valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_validation'
testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_test'





#############################################
# Extracts the patches from the image and 
# saves to path
#############################################
def imagePatchExtractor(imageSet, imagepath, finepath, outpath, mode):
	counter = 0
	index = 0
	
	fileIndex = 1
	if mode == 'Train':
		x_trainHandler = open(outTrainImgPath+'/'+'X_train_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
		y_trainHandler = open(outTrainImgPath+'/'+'Y_train_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
	elif mode == 'Val':
		x_valHandler = open(outValImgPath+'/'+'X_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
		y_valHandler = open(outValImgPath+'/'+'Y_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
	elif mode == 'Test':
		x_testHandler = open(outTestImgPath+'/'+'X_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
		y_testHandler = open(outTestImgPath+'/'+'Y_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
	
	imArray = np.array([])
	yLabels = np.array([])
	
	#counter = 0
	for city in sorted(os.listdir(imagepath)):
		for file in sorted(os.listdir(imagepath+'/'+city)):
			print counter
			if imageSet is not None and counter == imageSet: # Extact patches from a number of frames
				if mode == 'Train':
					#print '3'
					if not x_trainHandler.closed and not y_trainHandler.closed:
						#print '4'
						np.save(x_trainHandler, imArray)
						np.save(y_trainHandler, yLabels)
						x_trainHandler.close()
						y_trainHandler.close() 
						imArray = np.array([])
						yLabels = np.array([])
				elif mode == 'Val':
					if not x_valHandler.closed and not y_valHandler.closed:
						np.save(x_valHandler, imArray)
						np.save(y_valHandler, yLabels)
						x_valHandler.close()
						y_valHandler.close() 
						imArray = np.array([])
						yLabels = np.array([])
				elif mode == 'Test':
					if not x_testHandler.closed and not y_testHandler.closed:
						np.save(x_testHandler, imArray)
						np.save(y_testHandler, yLabels)
						x_testHandler.close()
						y_testHandler.close() 
				return

			image = io.imread(imagepath+'/'+city+'/'+file)
			h, w, c  = image.shape
			# load the annoated image
			labelImage = io.imread(finepath+'/'+city+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
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
					
					centerLabel = label[patchSize/2, patchSize/2]
					if centerLabel == 255:
						j += patchSize
						#counter += 1
						continue
					#try:
						#rename = file.replace(rawImagePattern, '{0:07d}'.format(counter) +'_'+rawImagePattern)
					im = np.array(croppedImage)
					if imArray.size == patchSize*patchSize*c:
						imArray = np.stack((imArray, im), axis=0)
						yLabels = np.append(yLabels, centerLabel)
					elif imArray.size == 0:
						imArray = im
						yLabels = np.append(yLabels, centerLabel)
					else:
						imArray = np.insert(imArray, index, im, axis=0)
						yLabels = np.append(yLabels, centerLabel)			

					if index == offset-1:
						if mode == 'Train':
							np.save(x_trainHandler, imArray)
							np.save(y_trainHandler, yLabels)
							fileIndex += 1
							x_trainHandler.close()
							y_trainHandler.close()
							# Reset the arrays for refill
							imArray = np.array([])
							yLabels = np.array([])

							x_trainHandler = open(outpath+'/'+'X_train_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							y_trainHandler = open(outpath+'/'+'Y_train_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							index = 0
							continue
						elif mode == 'Val':
						#if index == offset-1:
							np.save(x_valHandler, imArray)
							np.save(y_valHandler, yLabels)
							x_valHandler.close()
							y_valHandler.close()
							# Reset the arrays for refill
							imArray = np.array([])
							yLabels = np.array([])

							fileIndex += 1
							x_valHandler = open(outpath+'/'+'X_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							y_valHandler = open(outpath+'/'+'Y_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							index = 0
							continue
						elif mode == 'Test':
					#if index == offset-1:
							np.save(x_testHandler, imArray)
							np.save(y_testHandler, yLabels)
							fileIndex += 1
							x_testHandler.close()
							y_testHandler.close()
							# Reset the arrays for refill
							imArray = np.array([])
							yLabels = np.array([])

							x_testHandler = open(outpath+'/'+'X_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							y_testHandler = open(outpath+'/'+'Y_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
							index = 0
							continue
						#io.imsave(outpath+'/'+ labels.listLabels[centerLabel]+'/'+rename, croppedImage)
					#except Exception as e:
					#	print centerLabel
					#	raise e
					#	sys.exit(-1)
					index +=1
					j += patchSize
					#counter += 1
				
				i += patchSize
				# Leave the last row pixels (max patchsize pixels abandoned)
				if i > h-patchSize:
					break
				j = 0
	# Check if the file handlers are closed with the residual samples
	if mode == 'Train':
		if not x_trainHandler.closed and not y_trainHandler.closed:
			np.save(x_trainHandler, imArray)
			np.save(y_trainHandler, yLabels)
			x_trainHandler.close()
			y_trainHandler.close() 
			imArray = np.array([])
			yLabels = np.array([])
	elif mode == 'Val':
		if not x_valHandler.closed and not y_valHandler.closed:
			np.save(x_valHandler, imArray)
			np.save(y_valHandler, yLabels)
			x_valHandler.close()
			y_valHandler.close() 
			imArray = np.array([])
			yLabels = np.array([])
	elif mode == 'Test':
		if not x_testHandler.closed and not y_testHandler.closed:
			np.save(x_testHandler, imArray)
			np.save(y_testHandler, yLabels)
			x_testHandler.close()
			y_testHandler.close() 

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
	#folderCheck()
	
	# Extract Raw images
	print('Train set...')
	imagePatchExtractor(trainImageSet, trainImagePath, trainFinePath, outTrainImgPath, mode='Train')
	'''
	counter = 0
	for city in sorted(os.listdir(trainImagePath)):
		for file in sorted(os.listdir(trainImagePath+'/'+city)):
			if trainImageSet is not None and counter == trainImageSet:
				break
			image = io.imread(trainImagePath+'/'+city+'/'+file)
			counter += 1
	'''
	print('Validation Set...')
	imagePatchExtractor(valImageSet, valImagePath, valFinePath, outValImgPath, mode='Val')
	'''
	counter = 0
	for city in sorted(os.listdir(valImagePath)):
		for file in sorted(os.listdir(valImagePath+'/'+city)):
			if valImageSet is not None and counter == valImageSet:
				break
			image = io.imread(valImagePath+'/'+city+'/'+file)
			counter += 1
	'''
	print('Test set...')
	imagePatchExtractor(testImageSet, testImagePath, testFinePath, outTestImgPath, mode='Test')
	'''
	counter = 0
	for city in sorted(os.listdir(testImagePath)):
		for file in sorted(os.listdir(testImagePath+'/'+city)):
			if testImageSet is not None and counter == testImageSet:
				break
			image = io.imread(testImagePath+'/'+city+'/'+file)
			#image = image/255.0
			counter += 1
	'''
if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))