"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

'''
import numpy as np
from keras.activations import softmax
from keras.objectives import categorical_crossentropy

# init tests
samples=3
maxlen=4
vocab=5

y_pred_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
y_pred = K.variable(y_pred_n)
y_pred = softmax(y_pred)

y_true_n = np.random.random((samples,maxlen,vocab)).astype(K.floatx())
y_true = K.variable(y_true_n)
y_true = softmax(y_true)

# test 1 that it works the same as categorical_crossentropy with weights of one
weights = np.ones(vocab)

loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
np.testing.assert_almost_equal(loss_weighted,loss)
print('OK test1')


# test 2 that it works differen't than categorical_crossentropy with weights of less than one
weights = np.array([0.1,0.3,0.5,0.3,0.5])

loss_weighted=weighted_categorical_crossentropy(weights)(y_true,y_pred).eval(session=K.get_session())
loss=categorical_crossentropy(y_true,y_pred).eval(session=K.get_session())
np.testing.assert_array_less(loss_weighted,loss)
print('OK test2')

# same keras version as I tested it on?
import keras
assert keras.__version__.split('.')[:2]==['2', '0'], 'this was tested on keras 2.0.6 you have %s' % keras.__version
print('OK version')
'''