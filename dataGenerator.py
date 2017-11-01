import numpy as np
import os
import random
#import re
#from keras.callbacks import Callback
from keras import backend as K
#import keras.utils.np_utils.to_categorical
from keras.utils.np_utils import to_categorical
import labels
#from sklearn.preprocessing import StandardScaler 

class DataGenerator(object):
	"""
	DataGenerator Class:
	Yields the data in chunks of some size to reduce memory consumption
	Input:
			batchSize: The size of the batch to be yield in the fit generetor 
			patchSize: The row/column size of each image
			dataPath: The path to fetch the data from
			mode: Either 'Train', 'Val' or 'Test'
	"""
	def __init__(self, numClasses, batchSize, patchSize, trainPath, valPath, testPath):
		
		self.batchSize = batchSize
		self.patchSize = patchSize
		self.trainPath = trainPath
		self.valPath = valPath
		self.testPath = testPath
		self.classes = numClasses
		#self.mode = mode
		self.trainIndex = 0
		self.valIndex = 0
		self.testIndex = 0
		self.trainSetSize = 0
		self.valSetSize = 0
		self.testSetSize = 0
		self.mean = 0
		self.std = 0
		self.allTestClasses = np.array([])
		self.X_trainList = []
		self.Y_trainList = []
		self.X_valList = []
		self.Y_valList = []
		self.X_testList = []
		self.Y_testList = []
		self.loadData()
		self.datasetSize()
		self.getStats()

	'''
	Returns a batch of samples requested from the generator
	'''
	def getNextBatch(self, fileId, mode):
		if mode == 'Train':
				#index = self.trainIndex
			#for i in range(self.batchSize):
				#index = 0
				X_train = np.load(self.trainPath+'/'+fileId).astype(float)
				Y_train = np.load(self.trainPath+'/'+self.Y_trainList[self.X_trainList.index(fileId)])
				X, Y = self.shuffle(X_train, Y_train)
				X = np.true_divide(X, 255.)
				X_scaled = self.scale(X)
				X_scaled = np.reshape(X_scaled, (X_scaled.shape[0],self.patchSize,self.patchSize,3))
				#self.X_trainList.index(fileId)
				#while index < self.trainSetSize: 
				'''
				try:
						#Y_out = np.zeros((self.batchSize, self.classes))
						#Y_out[Y[index:(index+self.batchSize)]] = 1
						#print Y_out.shape
					return (X_scaled[index:(index+self.batchSize)], 
					to_categorical(Y[index:(index+self.batchSize)], self.classes))
				except:
					return (X_scaled[index::], to_categorical(Y[index::], self.classes))
				#index += self.batchSize
				'''
				return X_scaled, Y


		elif mode == 'Val':
			#for file in self.X_valList:
				#index = 0
				X_val = np.load(self.valPath+'/'+fileId)
				Y_val = np.load(self.valPath+'/'+self.Y_valList[self.X_valList.index(fileId)])
				X, Y = self.shuffle(X_val, Y_val)
				X = np.true_divide(X, 255.)
				X_scaled = self.scale(X)
				X_scaled = np.reshape(X_scaled, (X_scaled.shape[0],self.patchSize,self.patchSize,3))
				#while index < self.valSetSize: 
				#try:
					#yield X_scaled[index:(index+self.batchSize)], Y_val[index:(index+self.batchSize)]
				return X_scaled, Y_val
				#except :
					#yield X_scaled[index::], Y_val[index::]
				#index += self.batchSize

		elif mode == 'Test':
			#for file in self.X_testList:
				#index = 0
				X_test = np.load(self.testPath+'/'+fileId)
				Y_test = np.load(self.testPath+'/'+self.Y_testList[self.X_testList.index(fileId)])
				X, Y = self.shuffle(X_test, Y_test)
				X = np.true_divide(X, 255.)				
				self.allTestClasses = np.append(self.allTestClasses, Y)
				X_scaled = self.scale(X)
				X_scaled = np.reshape(X_scaled, (X_scaled.shape[0],self.patchSize,self.patchSize,3))
				#while index < self.testSetSize: 
				#	try:
				#		yield X_scaled[index:(index+self.batchSize)], Y_test[index:(index+self.batchSize)]
				#	except :
				#		yield X_scaled[index::], Y_test[index::]
				#	index += self.batchSize
				return X_scaled, Y_test
	
	def nextTrain(self):
		counter = 0
		while True:
			file = self.X_trainList[counter]
			counter = (counter+1) % len(self.X_trainList)			
			x, y = self.getNextBatch(file, mode='Train')		
			for cbatch in range(0, x.shape[0], self.batchSize):
				yield x[cbatch:(cbatch+self.batchSize),:,:,:], to_categorical(y[cbatch:(cbatch+self.batchSize)])
			
	def nextVal(self):
		counter = 0
		while True:
			file = self.X_valList[counter]
			counter = (counter+1) % len(self.X_valList)		
			x, y = self.getNextBatch(file, mode='Val')
			for cbatch in range(0,x.shape[0], self.batchSize):
				yield x[cbatch:(cbatch+self.batchSize),:,:,:], to_categorical(y[cbatch:(cbatch+self.batchSize)])					

	def nextTest(self):
		counter = 0
		while True:
			file = self.X_testList[counter]
			counter = (counter+1) % len(self.X_trainList)	
			x, y = self.getNextBatch(file, mode='Test')
			for cbatch in range(0, x.shape[0], self.batchSize):
				yield x[cbatch:(cbatch+self.batchSize),:,:,:], to_categorical(y[cbatch:(cbatch+self.batchSize)])
					
		
	'''
	Shuffle the samples to overcome overfitting with specific classes 
	'''
	def shuffle(self, x, y):
		s = np.arange(x.shape[0])
		np.random.shuffle(s)
		return x[s], y[s] 
	
	'''
	Function scale used for scaling the samples to get zero mean and unit variance
	'''
	def scale(self, x):
		x -= self.mean
		x = np.true_divide(x, self.std)
		return x
	
	'''
	Load the data from the file to a list, seperate the labels from
	the samples
	'''
	def loadData(self):
		for file in sorted(os.listdir(self.trainPath)):
			if file not in labels.listLabels:
				if file.startswith('X'):
					self.X_trainList.append(file)
				else:
					self.Y_trainList.append(file)

		for file in sorted(os.listdir(self.valPath)):
			if file not in labels.listLabels:
				if file.startswith('X'):
					self.X_valList.append(file)
				else:
					self.Y_valList.append(file)

		for file in sorted(os.listdir(self.testPath)):
			if file not in labels.listLabels:
				if file.startswith('X'):
					self.X_testList.append(file)
				else:
					self.Y_testList.append(file)
				
			

	'''
	Calculate the mean and variance from samples from 
	the train set to standardize the datasets, we take randomly
	some samples from the train set to make samplewise normalization
	reducing the computation requirements for the operation
	'''
	def getStats(self):
		X_sample = np.load(self.trainPath+'/'+random.choice(self.X_trainList))
		X_sample = np.true_divide(X_sample, 255.)
		self.mean = np.mean(X_sample, axis=0)
		self.std = np.std(X_sample, axis=0)+K.epsilon() # Add a small constant to prevent division with zero
		return
	
	'''
	Get the number of samples from each set(train, validation, test) from the 
	partitioned files
	'''
	def datasetSize(self):
		for file in self.Y_trainList:
			#print file
			X = np.load(self.trainPath+'/'+file)
			#print X.shape
			self.trainSetSize += X.shape[0]
		for file in self.Y_valList:
			X = np.load(self.valPath+'/'+file)
			#print X.shape
			self.valSetSize += X.shape[0]
		for file in self.Y_testList:
			X = np.load(self.testPath+'/'+file)
			self.testSetSize += X.shape[0]
		return
	
	'''
	Returns the size of each set selected by mode variable
	'''
	def getSize(self, mode):
		if mode == 'Train':
			return self.trainSetSize
		elif mode == 'Val':
			return self.valSetSize
		elif mode == 'Test':
			return self.testSetSize
	'''
	Returns the yLabels from test set: Useful to extract confusion 
	matrices and F1-scores
	'''
	def getClasses(self):
		return self.allTestClasses