import numpy as np
import os
import random
import re
from keras.callbacks import Callback
from keras import backend as K
import labels
#from sklearn.preprocessing import StandardScaler 

class DataGenerator(Callback):
	"""
	DataGenerator Class:
	Yields the data in chunks of some size to reduce memory consumption
	Input:
			batchSize: The size of the batch to be yield in the fit generetor 
			patchSize: The row/column size of each image
			dataPath: The path to fetch the data from
			mode: Either 'Train', 'Val' or 'Test'
	"""
	def __init__(self, batchSize, patchSize, trainPath, valPath, testPath):
		
		self.batchSize = batchSize
		self.patchSize = patchSize
		self.trainPath = trainPath
		self.valPath = valPath
		self.testPath = testPath
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

	def getNextBatch(self, mode):
		if mode == 'Train':
			for file in self.X_trainList:
				index = 0
				X_train = np.load(self.trainPath+'/'+file)
				Y_train = np.load(self.trainPath+'/'+self.Y_trainList[self.X_trainList.index(file)])
				X, Y = self.shuffle(X_train, Y_train)
				X_scaled = self.scale(X)
				while index < self.trainSetSize: 
					try:
						yield X_scaled[index:(index+self.batchSize)], Y[index:(index+self.batchSize)]
					except:
						yield X_scaled[index::], Y[index::]
					index += self.batchSize

		elif mode == 'Val':
			for file in self.X_valList:
				index = 0
				X_val = np.load(self.valPath+'/'+file)
				Y_val = np.load(self.valPath+'/'+self.Y_valList[self.X_valList.index(file)])
				X, Y = self.shuffle(X_val, Y_val)
				X_scaled = self.scale(X)
				while index < self.valSetSize: 
					try:
						yield X_scaled[index:(index+self.batchSize)], Y_val[index:(index+self.batchSize)]
					except :
						yield X_scaled[index::], Y_val[index::]
					index += self.batchSize

		elif mode == 'Test':
			for file in self.X_testList:
				index = 0
				X_test = np.load(self.testPath+'/'+file)
				Y_test = np.load(self.testPath+'/'+self.Y_testList[self.X_testList.index(file)])
				X, Y = self.shuffle(X_test, Y_test)
				np.append(self.allTestClasses, Y)
				X_scaled = self.scale(X)
				while index < self.testSetSize: 
					try:
						yield X_scaled[index:(index+self.batchSize)], Y_test[index:(index+self.batchSize)]
					except :
						yield X_scaled[index::], Y_test[index::]
					index += self.batchSize
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
		x /= self.std
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
		self.mean = np.mean(X_sample, axis=0)
		self.std = np.std(X_sample, axis=0)+K.epsilon() # Add a small constant to prevent division with zero
		return
	
	'''
	Get the number of samples from each set(train, validation, test) from the 
	partitioned files
	'''
	def datasetSize(self):
		for file in self.Y_trainList:
			X = np.load(self.trainPath+'/'+file)
			self.trainSetSize += X.shape[0]
		for file in self.Y_valList:
			X = np.load(self.valPath+'/'+file)
			self.valSetSize += X.shape[0]
		for file in self.Y_testList:
			X = np.load(self.testPath+'/'+file)
			self.testSetSize += X.shape[0]
		return
	'''

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
		return allTestClasses