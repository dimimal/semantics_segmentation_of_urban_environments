#
# Callback for testing the model on every epoch
# Usefull to plot the testing curve
import numpy as np

class TestCallback(Callback):
    def __init__(self, epochs, testgenerator):
        #self.test_data = test_data
        self.score = np.zeros([epochs,2])
        self.generator = testgenerator

    def on_epoch_end(self, epoch, logs={}):
        #x, y = self.test_data
        self.score[epoch-1][:] = model.evaluate_generator(self.generator,use_multiprocessing=True,verbose=0)
        #loss, acc = self.model.evaluate_evaluate(x, y, verbose=0)
        print('\nTesting loss: {}, acc: {}\n'.format(score[epoch-1][0], score[epoch-1][1]))