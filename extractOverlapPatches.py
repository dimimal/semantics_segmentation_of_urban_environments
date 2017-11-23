#!/usr/bin/python					

import os
import time
from sklearn.feature_extraction import image 
from skimage import io
import re
import numpy as np

patchSize = 16
stride = 4
rawImagePattern = 'leftImg8bit.png'
finePattern = 'gtFine_labelTrainIds.png'
# Validation set Paths
valImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_validation'
outValImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/validation_set_'+str(patchSize)

# Test set Paths
testImagePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/resized_test'
outTestImgPath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/leftImg8bit/test_set_'+str(patchSize)

valFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_validation'
testFinePath = '/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/resized_test'

xValFileObj = open(outValImgPath+'/'+'X_val_set_'+str(patchSize)+'_'+'%04d.npz'%(1), 'wb')
yValFileObj = open(outValImgPath+'/'+'Y_val_set_'+str(patchSize)+'_'+'%04d.npz'%(1), 'wb')

xTestFileObj = open(outTestImgPath+'/'+'X_test_set'+str(patchSize)+'_'+'%04d.npz'%(1), 'wb')
yTestFileObj = open(outTestImgPath+'/'+'Y_test_set'+str(patchSize)+'_'+'%04d.npz'%(1), 'wb')

valImageSet = 150
testImageSet = None

# ##############################################################
# Extracts patches with overlap, to select stride change 
# stride variable
# ##############################################################
def stridedExtractor(imageSet, imagepath, finepath, outpath, mode):
	counter = 0	
	index = 0
	offset = 9999999
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
	for file in sorted(os.listdir(imagepath)):
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

		image = io.imread(imagepath+'/'+file)
		h, w, c  = image.shape
		# load the annoated image
		labelImage = io.imread(finepath+'/'+re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
		counter += 1
		i = 0
		j = 0

		while i < h :
	 		while j < w:
				# Check boundaries
	 			if j > w or j > w-patchSize:
	 				croppedImage = image[i:(i+patchSize),(w-patchSize):,:]
	 				croppedFineImage = labelImage[i:i+patchSize,(w-patchSize):]
	 			else:
	 				croppedImage = image[i:(i+patchSize),j:(j+patchSize),:]
	 				croppedFineImage = labelImage[i:(i+patchSize),j:(j+patchSize)]
	 			
				# Extract the center pixel(Label)
				centerLabel = croppedFineImage[patchSize//2, patchSize//2]
				if centerLabel == 255:
					j += stride
					continue
				# Change the rawImagePattern from file name to index_rawImagePattern
				#rename = file.replace(rawImagePattern, str(counter)+'_'+rawImagePattern)
				
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
				index += 1
	 			j += stride	 		
	 		i += stride
			# Leave the last row pixels (max patchsize-1 pixels abandoned)
	 		if i > h-patchSize:
	 			break
	 		j = 0


def extractLabels(rawPatches, finePatches):
	Labels = np.array([])
	images = np.array([])
	#print(rawPatches.shape, finePatches.shape)
	for i in range(finePatches.shape[0]):
		#print(finePatches.shape)
		#print(Labels.shape)
		#print()
		#print(i)
		if finePatches[i,finePatches.shape[1]/2, finePatches.shape[2]/2] == 255:
			#rawPatches = np.delete(rawPatches, i, axis=0)
			continue
		if images.size == patchSize*patchSize*3:
			images = np.stack((images, rawPatches[i]), axis=0)
		elif images.size == 0:
			images = rawPatches[i]
		else:
			images = np.append(images, rawPatches[i])


		Labels = np.append(Labels, finePatches[i,finePatches.shape[1]/2, finePatches.shape[2]/2])
		#print(images.shape)
		'''
		if Labels[i] == 255:
			Labels = np.delete(Labels, i, axis=0)
			rawPatches = np.delete(rawPatches, i, axis=0)
		'''
	return images, Labels

def imagePatchExtractor(imageSet, imagepath, finepath, outpath, mode):
	counter = 0
	index = 0
	offset = 3000 # how many samples per file
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
	#for city in sorted(os.listdir(imagepath)):
	for file in sorted(os.listdir(imagepath)):
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
				
				centerLabel = label[patchSize/2, patchSize/2]
				if centerLabel == 255:
					j += patchSize
					#counter += 1
					continue

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

def main():
	counter = 0
	print('Validation....')
	'''
	for city in sorted(os.listdir(valImagePath)):
		for file in sorted(os.listdir(valImagePath+'/'+city)):
			if valImageSet is not None and counter == valImageSet:
				break
			im = io.imread(valImagePath+'/'+city+'/'+file)
			labelImage = io.imread(valFinePath+'/'+city+'/'+ re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			patches = image.extract_patches_2d(im, (patchSize, patchSize))
			labelPatches = image.extract_patches_2d(labelImage, (patchSize, patchSize))
			patches, yLabels = extractLabels(patches, labelPatches)
			np.save(xValFileObj, patches)
			np.save(yValFileObj, yLabels)
			fileIndex += 1
			xValFileObj = open(outValImgPath+'/'+'X_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			yValFileObj = open(outValImgPath+'/'+'Y_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			print(counter)
			counter += 1
	
	for file in sorted(os.listdir(valImagePath)):
			if valImageSet is not None and counter == valImageSet:
				break
			im = io.imread(valImagePath+'/'+file)
			labelImage = io.imread(valFinePath+'/'+ re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			patches = image.extract_patches(im, (patchSize, patchSize, 3), extraction_step=(9,9,3))
			labelPatches = image.extract_patches(labelImage, (patchSize, patchSize), extraction_step=(9,9))
			print(patches.shape)
			print(labelPatches.shape) 
			patches, yLabels = extractLabels(patches, labelPatches)
			np.save(xValFileObj, patches)
			np.save(yValFileObj, yLabels)
			fileIndex += 1
			xValFileObj = open(outValImgPath+'/'+'X_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			yValFileObj = open(outValImgPath+'/'+'Y_val_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			print(counter)
			counter += 1

	
	fileIndex = 1
	counter = 0
	print('Test....')
	for city in sorted(os.listdir(testImagePath)):
		for file in sorted(os.listdir(testImagePath+'/'+city)):
			if testImageSet is not None and counter == testImageSet:
				break
			im = io.imread(valImagePath+'/'+city+'/'+file)
			labelImage = io.imread(valFinePath+'/'+city+'/'+ re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			patches = image.extract_patches_2d(im, (patchSize, patchSize))
			labelPatches = image.extract_patches_2d(labelImage, (patchSize, patchSize))
			patches, yLabels = extractLabels(patches, labelPatches)
			np.save(xTestFileObj, patches)
			np.save(yTestFileObj, yLabels)
			fileIndex += 1
			xValFileObj = open(outValImgPath+'/'+'X_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			yValFileObj = open(outValImgPath+'/'+'Y_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			print(counter)
			counter += 1
	fileIndex = 1
	counter = 0
	print('Test....')
	for file in sorted(os.listdir(testImagePath)):
			if testImageSet is not None and counter == testImageSet:
				break
			im = io.imread(testImagePath+'/'+file)
			labelImage = io.imread(testFinePath+'/'+ re.findall('\w+_\d+_\d+_', file)[0]+finePattern)
			
			patches = image.extract_patches_2d(im, (patchSize, patchSize))
			labelPatches = image.extract_patches_2d(labelImage, (patchSize, patchSize))
			patches, yLabels = extractLabels(patches, labelPatches)
			
			stridedExtractor(im, labelImage, file, mode='Val')
			np.save(xTestFileObj, patches)
			np.save(yTestFileObj, yLabels)
			fileIndex += 1
			xValFileObj = open(outValImgPath+'/'+'X_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			yValFileObj = open(outValImgPath+'/'+'Y_test_set_'+str(patchSize)+'_'+'%04d.npz'%(fileIndex), 'wb')
			print(counter)
			counter += 1
	'''
	stridedExtractor(valImageSet, valImagePath, valFinePath, outValImgPath,mode='Val')
	stridedExtractor(testImageSet, testImagePath, testFinePath, outTestImgPath, mode='Test')
	
if __name__ == '__main__':
	start_time = time.time()
	main()
	print("--- %s seconds ---" % (time.time() - start_time))