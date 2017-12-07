from mpool import m_maxpool2d, m_maxpool3d
from keras.engine.topology import Layer as Layer_inh
from keras import backend as K
#from theano import tensor as T
from keras.layers.pooling import _Pooling2D, _Pooling3D
import tensorflow as tf

class MemoMaxPooling2D(_Pooling2D):
    def __init__(self, pool_size=(2, 2), strides=None, padding='valid',
                 dim_ordering=K.image_data_format(), **kwargs):
        print dim_ordering
        super(MemoMaxPooling2D, self).__init__(pool_size, strides, padding,
                                               dim_ordering, **kwargs)

    def _pooling_function(self, inputs, pool_size, strides,
                          padding, dim_ordering):
        output, ind = m_maxpool2d(inputs, pool_size, strides,
                                  padding, dim_ordering)
        self.ind = ind
        return output


class MemoUpSampling2D(Layer_inh):
    def __init__(self, pooling_layer, size=(2, 2), shape=None, dim_ordering=K.image_data_format(), **kwargs):
        self.size = tuple(size)
        assert dim_ordering in {'channels_first', 'channels_last'}, 'dim_ordering must be in {channels_first, channels_last}'
        self.dim_ordering = dim_ordering
        # self.input_spec = [InputSpec(ndim=4)]
        self.ind = pooling_layer.ind
        assert shape is None or len(shape) == 2, 'shape should be 2D vector'
        self.shape = shape or pooling_layer.ind.shape[-2:]
        super(MemoUpSampling2D, self).__init__(**kwargs)

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'channels_first':
            self.shape = self.shape or self.size * input_shape[:-2]
            return (input_shape[0],
                    input_shape[1],
                    self.shape[0],
                    self.shape[1])
        elif self.dim_ordering == 'channels_last':
            self.shape = self.shape or self.size * input_shape[1:-1]
            return (input_shape[0],
                    self.shape[0],
                    self.shape[1],
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        # TODO: implement for dim_ordering='channels_last'
        if self.dim_ordering=='channels_last':
            img = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
            padded = K.zeros((img.shape[0], img.shape[1], self.shape[0], self.shape[1]))
            padded = T.set_subtensor(padded[:,:img.shape[1], :img.shape[2], :], img)
            #padded = tf.scatter_nd(indices, updates, shape)
            return K.switch(self.ind, padded, K.zeros_like(padded))
        elif self.dim_ordering=='channels_first':
            '''
            img = K.resize_images(x, self.size[0], self.size[1], self.dim_ordering)
            padded = T.zeros((img.shape[0], img.shape[1], self.shape[0], self.shape[1]))
            padded = T.set_subtensor(padded[:, :, :img.shape[2], :img.shape[3]], img)
            return T.switch(self.ind, padded, T.zeros_like(padded))
            '''
            raise Exception('Channels_first is not supported')

    def get_config(self):
        config = {'size': self.size}
        base_config = super(MemoUpSampling2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))