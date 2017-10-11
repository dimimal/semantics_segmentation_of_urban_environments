# Callback for testing the model on every epoch
# Usefull to plot the testing curve

import numpy as np
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, epochs, testgenerator, batchsize, testSetSize):
        self.score = np.zeros([epochs,2])
        self.generator = testgenerator
        self.batchsize = batchsize
        self.set = testSetSize

    def on_epoch_end(self, epoch, logs={}):           
        self.score[epoch,:] = self.model.evaluate_generator(self.generator,steps=self.set//self.batchsize,use_multiprocessing=True)
        print('\nTesting loss: {}, acc: {}\n'.format(self.score[epoch,0], self.score[epoch,1]))