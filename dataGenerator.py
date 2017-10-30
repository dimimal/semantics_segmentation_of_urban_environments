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
		self.mean = None
		self.std = None
		self.loadData()
		self.getStats()

	def getNextBatch(self, mode):
		if mode == 'Train':
			index = 0
			for file in self.trainList:
				if file.startswith('X'):
					x_train = np.load(self.trainPath+'/'+file)
					np.load(file) 
		elif mode == 'Val':
			for file in self.trainList:
				if file.startswith('X'):
					np.load(self.trainPath+'/'+file)
					np.load(file) 
		elif mode == 'Test':
			for file in self.trainList:
				if file.startswith('X'):
					np.load(self.trainPath+'/'+file)
					np.load(file) 

	def shuffle(self, x):
		s = np.arange(x.shape[0])
		np.random.shuffle(s)


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
		#X_sample = np.load
				#sample = re.findall('X_\w+_\w+_\d+_d+.npy', file)[0]
				'''
				if file.startswith('X'):
					self.X_sample = np.load(self.path+'/'+file)
					file.lstrip('X')
					self.Y_sample = np.load(self.path+'/'+'Y'+file)
				'''

	'''
	Calculate the mean and variance from samples from 
	the train set to standardize the datasets, we take randomly
	some samples from the train set to make samplewise normalization
	'''
	def getStats(self):
		X_sample = np.load(self.trainPath+'/'+random.choice(self.X_trainList))
		self.mean = np.mean(X_sample, axis=0)
		self.std = np.std(X_sample, axis=0)
		return
	