import numpy as np
import os
import re
from keras.callbacks import Callback
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
	def __init__(self, batchSize, patchSize, dataPath, mode):
		
		self.batchSize = batchSize
		self.patchSize = patchSize
		self.path = dataPath

	def getNextBatch(self):
		pass

	def loadData(self):
		for file in sorted(os.listdir(self.path)):
			if file not in labels.listLabels:
				sample = re.findall('\w+_\w+_\w+_\d+_d+.npy', file)[0]
				if sample.startswith('X'):



	def normalize(self):
		pass	