#
# Callback for testing the model on every epoch
# Usefull to plot the testing curve
import numpy as np
from keras.callbacks import Callback

class TestCallback(Callback):
    def __init__(self, epochs, testgenerator, batchsize, testSetSize):
        #self.test_data = test_data
        self.score = np.zeros([epochs,2])
        self.generator = testgenerator
        self.batchsize = batchsize
        self.set = testSetSize

    def on_epoch_end(self, epoch, logs={}):
             
        #x, y = self.test_data
        self.score[epoch][:] = self.model.evaluate_generator(self.generator,steps=self.set//self.batchsize,use_multiprocessing=True)
        #loss, acc = self.model.evaluate_evaluate(x, y, verbose=0)
        #print('Epoch', epoch)
        print('\nTesting loss: {}, acc: {}\n'.format(self.score[epoch][0], self.score[epoch][1]))