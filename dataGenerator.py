import numpy as np
import os
import random
from keras.preprocessing.image import *
from keras.applications.imagenet_utils import preprocess_input
from keras import backend as K
from keras.utils.np_utils import to_categorical
import labels
from sklearn.utils import shuffle
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
    def __init__(self, 
                 numClasses, 
                 batchSize, 
                 img_rows, 
                 img_cols, 
                 trainPath, 
                 valPath, 
                 testPath,
                 shear_range=0., 
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='constant',
                 cval=0.,
                 prob=0.2, 
                 label_cval=19,
                 zoom_maintain_shape=True,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 normalize=True):

        self.channels               = 3
        self.batchSize              = batchSize
        self.img_rows               = img_rows
        self.img_cols               = img_cols
        self.trainPath              = trainPath
        self.valPath                = valPath
        self.testPath               = testPath
        self.classes                = numClasses
        self.normalize              = normalize
        self.shear_range            = shear_range
        self.channel_shift_range    = channel_shift_range       
        self.fill_mode              = fill_mode
        self.rotation_range         = rotation_range
        self.width_shift_range      = width_shift_range
        self.height_shift_range     = height_shift_range
        self.horizontal_flip        = horizontal_flip
        self.vertical_flip          = vertical_flip 
        self.zoom_maintain_shape    = zoom_maintain_shape
        self.cval                   = cval
        self.label_cval             = label_cval
        self.prob                   = prob
        self.dataAugMode            = False
        self.trainIndex             = 0
        self.valIndex               = 0
        self.testIndex              = 0
        self.trainSetSize           = 0
        self.valSetSize             = 0
        self.testSetSize            = 0
        self.mean                   = np.array([])
        self.std                    = np.array([])
        self.allTestClasses         = np.array([])
        self.X_trainList            = []
        self.Y_trainList            = []
        self.X_valList              = []
        self.Y_valList              = []
        self.X_testList             = []
        self.Y_testList             = []
        self.loadData()
        self.datasetSize()
        self.getStats()
        
        # Get data format, if channels first raise error
        data_format    = K.image_data_format()
        
        if data_format == 'channels_first':
            raise Exception('This works only with channels last')

        if data_format == 'channels_last':
            self.channel_index  = 3
            self.row_index      = 1
            self.col_index      = 2

        # Set zoom range if available
        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)
        
        # Check if data augmentation is on
        if horizontal_flip or vertical_flip or channel_shift_range \
        or width_shift_range or height_shift_range:
            self.dataAugMode = True
            
    '''
    Returns a batch of samples requested from the generator
    '''
    def getNextBatch(self, fileId, mode):
        if mode == 'Train':
            X_train = np.load(self.trainPath+'/'+fileId).astype('float32')
            Y_train = np.load(self.trainPath+'/'+self.Y_trainList[self.X_trainList.index(fileId)]).astype('float32')
            #X_train = np.load(self.trainPath+'/'+'X_train_set_512_0005.npz').astype('float32')
            #Y_train = np.load(self.trainPath+'/'+'Y_train_set_512_0005.npz')
            X_train, Y_train = self.shuffle(X_train, Y_train)
            if self.normalize:
                X_train = self.standardize(X_train)
            return X_train, Y_train


        elif mode == 'Val':
            X_val = np.load(self.valPath+'/'+fileId, mmap_mode='r').astype('float32')
            Y_val = np.load(self.valPath+'/'+self.Y_valList[self.X_valList.index(fileId)]).astype('float32')
            
            X_val, Y_val = self.shuffle(X_val, Y_val) # Shuffle data
            if self.normalize:  
                X_val = self.standardize(X_val)
            return X_val, Y_val

        elif mode == 'Test':
            X_test = np.load(self.testPath+'/'+fileId, mmap_mode='r').astype('float32')
            Y_test = np.load(self.testPath+'/'+self.Y_testList[self.X_testList.index(fileId)]).astype('float32')
            if self.normalize:
                X_test = self.standardize(X_test)
            return X_test, Y_test
        raise Exception('Wrong mode given::', mode)

    def nextTrain(self):
        counter = 0
        while True:
            file = self.X_trainList[counter]
            counter = (counter+1) % len(self.X_trainList)           
            x, y = self.getNextBatch(file, mode='Train')
            x = np.squeeze(x)
            y = np.squeeze(y)
            #y = self.oneHotEncode(y)
            #y = to_categorical(y.flatten(), self.classes).reshape((y.shape,self.img_rows,self.img_cols, self.classes)) 
            #y = np.reshape(y, (0, self.img_rows, self.img_cols))
            for cbatch in range(0, x.shape[0], self.batchSize):
                if x[cbatch:(cbatch+self.batchSize)].shape[0] != y[cbatch:(cbatch+self.batchSize)].shape[0]:
                    continue
                '''
                try:
                    
                    if random.random() < self.prob:
                        x[cbatch], y[cbatch, :] = self.random_transform(x[cbatch], y[cbatch].reshape((self.img_rows,self.img_cols)))                   
                    
                    if y[cbatch].size == 0:
                        continue
                    #print(y[cbatch:(cbatch+self.batchSize)].shape, cbatch)            
                    yield (x[cbatch:(cbatch+self.batchSize),:,:,:],#.reshape((self.batchSize,self.img_rows, self.img_cols,self.classes)))#y[cbatch:(cbatch+self.batchSize)]) 
                    np.expand_dims(to_categorical(y[cbatch:(cbatch+self.batchSize)].flatten(), self.classes)
                                    .reshape((self.img_rows, self.img_cols,self.classes)), axis=0))
                except:
                    #print(x[cbatch].shape, y[cbatch].shape)
                    continue
                '''
                yield (x[cbatch:(cbatch+self.batchSize),:,:,:], self.oneHotEncode(y[cbatch:(cbatch+self.batchSize)]))
                
    def nextVal(self):
        counter = 0
        while True:
            file = self.X_valList[counter]
            counter = (counter+1) % len(self.X_valList)     
            x, y = self.getNextBatch(file, mode='Val')
            #y = self.oneHotEncode(y)
            x = np.squeeze(x)
            y = np.squeeze(y)
            for cbatch in range(0,x.shape[0], self.batchSize):
                if x[cbatch:(cbatch+self.batchSize)].shape[0] != y[cbatch:(cbatch+self.batchSize)].shape[0]:
                    continue
                yield (x[cbatch:(cbatch+self.batchSize),:,:,:], self.oneHotEncode(y[cbatch:(cbatch+self.batchSize)]))                  

    def nextTest(self):
        counter = 0
        while True:
            file = self.X_testList[counter]
            counter = (counter+1) % len(self.X_testList)    
            x, y = self.getNextBatch(file, mode='Test')
            #y = self.oneHotEncode(y)            
            x = np.squeeze(x)
            y = np.squeeze(y)
            for cbatch in range(0, x.shape[0], self.batchSize):
                if x[cbatch:(cbatch+self.batchSize)].shape[0] != y[cbatch:(cbatch+self.batchSize)].shape[0]:
                    continue
                yield (x[cbatch:(cbatch+self.batchSize),:,:,:], self.oneHotEncode(y[cbatch:(cbatch+self.batchSize)]))
    
    '''
    Gets as input the dense ylabel matrix and returns the one 
    hot encoding of indexed labels: input-> y[n_sample,n_feats]
    '''
    def oneHotEncode(self, y):
        '''
        new_y = np.empty((y.shape[0],self.img_rows,self.img_cols,self.classes), dtype='uint8')
        for index,value in np.ndenumerate(y):
            new_y[index[0],index[1]/self.img_rows,index[1]%self.img_cols] = to_categorical(y[index[0],index[1]],\
                num_classes=self.classes)
        return new_y
        '''
        #print(y.shape)
        rows = y.shape[0]
        #if rows == 0:
        #    rows = 1
        y    = y.flatten()
        y    = to_categorical(y, self.classes)
        y    = np.reshape(y, (rows, self.img_rows, self.img_cols, self.classes))
        return y

    '''
    Shuffle the samples to overcome overfitting with specific classes 
    '''
    def shuffle(self, x, y):
        s = np.arange(x.shape[0])
        np.random.shuffle(s)
        return x[s], y[s] 
    
    '''
    Function used for scaling the samples to get zero mean and unit variance
    '''
    def standardize(self, X):
        X -= self.mean
        X  = np.true_divide(X, self.std)
        return X
    
    def fileShuffle(self, x ,y):
        return shuffle(x,y)
    '''
    Load the data from the file to a list, seperate the labels from
    the samples
    '''
    def loadData(self):
        for file in sorted(os.listdir(self.trainPath)):
            if file not in labels.listLabels:
                if file.startswith('X'):
                    self.X_trainList.append(file)
                elif file.startswith('Y'):
                    self.Y_trainList.append(file)
        #self.X_trainList, Y_trainList = self.fileShuffle(self.X_trainList, self.Y_trainList)
        
        for file in sorted(os.listdir(self.valPath)):
            if file not in labels.listLabels:
                if file.startswith('X'):
                    self.X_valList.append(file)
                elif file.startswith('Y'):
                    self.Y_valList.append(file)

        for file in sorted(os.listdir(self.testPath)):
            if file not in labels.listLabels:
                if file.startswith('X'):
                    self.X_testList.append(file)
                elif file.startswith('Y'):
                    self.Y_testList.append(file)
    '''
    Method which keeps the labels of the test samples to be used for
    prediction
    ''' 
    def computeTestClasses(self):
        for file in self.Y_testList:
            y = np.load(self.testPath+'/'+file).astype('float32')
            y = self.oneHotEncode(np.reshape(y, (y.shape[0], self.img_rows*self.img_cols)))
            if self.allTestClasses.size == 0:
                self.allTestClasses = y
            else:
                self.allTestClasses = np.concatenate((self.allTestClasses, y), axis=0)  
        return
    

    '''
    Calculate the mean and variance from samples from 
    the train set to standardize the datasets, we take randomly
    some samples from the train set to make samplewise normalization
    reducing the computation requirements for the operation
    '''
    def getStats(self):
        X_sample = np.load(self.trainPath+'/'+random.choice(self.X_trainList))
        self.mean = np.mean(X_sample, axis=0, keepdims=True)
        self.std = 255.
        return
    
    '''
    Get the number of samples from each set(train, validation, test) from the 
    partitioned files
    '''
    def datasetSize(self):
        for file in self.Y_trainList:
            Y = np.load(self.trainPath+'/'+file)
            self.trainSetSize += Y.shape[0]
        for file in self.Y_valList:
            Y = np.load(self.valPath+'/'+file)
            self.valSetSize += Y.shape[0]
        for file in self.Y_testList:
            Y = np.load(self.testPath+'/'+file)
            self.testSetSize += Y.shape[0]
        
        print('---------------------------------------')
        print(' Train Set Size      ----->  {}  '.format(self.trainSetSize))
        print('----------------------')
        print(' Validation Set Size ----->  {}  '.format(self.valSetSize))      
        print('---------------------------------------')
        print(' Test Set Size       ----->  {}  '.format(self.testSetSize))
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
    def getClasses(self, index):
        return self.allTestClasses[:index]

    # Random trasformation
    def random_transform(self, x, y):
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1
        y = np.expand_dims(y, axis=-1)

        assert x.shape[img_row_index] == y.shape[img_row_index] and x.shape[img_col_index] == y.shape[
            img_col_index], 'DATA ERROR: Different shape of data and label!\n\
                            data shape: {}, label shape: {}'.format(x.shape, y.shape)

        # use composition of homographies to generate final transform that
        # needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * \
                np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            # * x.shape[img_row_index]
            tx = np.random.uniform(-self.height_shift_range,
                                   self.height_shift_range)
        else:
            tx = 0

        if self.width_shift_range:
            # * x.shape[img_col_index]
            ty = np.random.uniform(-self.width_shift_range,
                                   self.width_shift_range)
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(
                self.zoom_range[0], self.zoom_range[1], 2)
        
        if self.zoom_maintain_shape:
            zy = zx
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(
            np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        
        #print(x.shape, y.shape)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        #print(transform_matrix)
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode='constant', cval=self.label_cval)

        if self.channel_shift_range != 0:
            x = random_channel_shift(
                x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)
        y = np.squeeze(y).flatten()
        #y = np.reshape(y, (1, y.shape[0]))
        #print(y.shape, np.squeeze(y,axis=-1).shape)
        return x, y