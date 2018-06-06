"""
A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
"""
from keras import backend as K
import tensorflow as tf


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

# Use this loss function with median frequency coefficients weights
# for class balance
def weighted_loss(num_classes, coefficients, head=None):
    
    coefficients = tf.constant(coefficients)
    num_classes = tf.constant(num_classes)

    def loss(labels, logits):
        with tf.name_scope('loss_1'):
            logits = tf.reshape(logits, (-1, num_classes))
            #print(tf.shape(logits))
            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon
            # consturct one-hot label array
            labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))
            softmax = tf.nn.softmax(logits)
            
            #softmax = logits    # Softmax already applied?? 
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), coefficients), reduction_indices=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            #print('cross :: {}'.format(cross_entropy_mean))

            tf.add_to_collection('losses', cross_entropy_mean)
            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
            #loss = cross_entropy_mean
        return loss
    return loss