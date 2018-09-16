import sys
from keras.models import Model, Sequential, model_from_json
from keras.layers import Input, Dense, Flatten, Activation, Reshape
from keras.layers import Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Lambda, core, Add, Concatenate, GlobalAveragePooling2D
from bilinearUpsampling import BilinearUpSampling2D
from lib.crfasrnn_keras.src.crfrnn_layer import CrfRnnLayer
from keras.activations import softmax
import keras.backend as K

def CRFRNN(unary_model, image_shape=(512,512,3), unary_input=(512,512,20), num_classes=20, theta_a=160.,\
    theta_b=90., theta_g=3., iters=5):
    """Implements CRF-RNN post processing unit along with SD_CNN or BD_CNN
    """

    # TODO: Pass CRF Hyperparameters
    if unary_model == 'sdcnn':
        front_model = SD_CNN(crf=True)
    elif  unary_model == 'bdcnn':
        front_model = bilinear_CNN(crf=True)
    else:
        raise Exception('Unknown CNN network {}'.format(unary_model))

    input_1     = Input(shape=(image_shape))
    input_2     = Input(shape=(unary_input))
    output      = CrfRnnLayer(image_dims=(image_shape[0], image_shape[1]),
                         num_classes=num_classes,
                         theta_alpha=theta_a,
                         theta_beta=theta_b,
                         theta_gamma=theta_g,
                         num_iterations=iters,
                         name='crfrnn')([input_2, input_1])
    #
    front_model = Model(inputs=front_model.input, outputs=front_model.output)
    top_model   = Model(inputs=[input_2, input_1], outputs=output)
    model       = Model(inputs=front_model.input, outputs=top_model([front_model.output, front_model.input]))
    
    return model



def bilinear_CNN(input_shape=(512,512,3), num_classes=20, crf=False):
    #input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)

    #
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_1')(inputs)
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_1')(x)
    #
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_3')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_2')(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_5')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_3')(x)
    #
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal', name='Conv_7')(x)
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal', name='Conv_8')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_4')(x)
    #
    atrous_1 = Conv2D(256, (3,3), dilation_rate=(3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_1')(x)
    atrous_1 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_2')(atrous_1)
    atrous_1 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_3')(atrous_1)
    #
    atrous_2 = Conv2D(256, (3,3), dilation_rate=(6,6), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_1')(x)
    atrous_2 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_2')(atrous_2)
    atrous_2 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_3')(atrous_2)
    #   
    atrous_3 = Conv2D(256, (3,3), dilation_rate=(9,9), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_1')(x)
    atrous_3 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_2')(atrous_3)
    atrous_3 = Conv2D(128, (1,1), padding='valid',  activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_3')(atrous_3)
    #
    atrous_4 = Conv2D(256, (3,3), dilation_rate=(12,12), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_1')(x)
    atrous_4 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_2')(atrous_4)
    atrous_4 = Conv2D(128, (1,1), padding='valid',  activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_3')(atrous_4)
    #
    atrous_5 = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_1')(x)
    atrous_5 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_2')(atrous_5)
    atrous_5 = Conv2D(128, (1,1), padding='valid', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_3')(atrous_5)
    x = Add(name='Fusion')([atrous_1, atrous_2, atrous_3, atrous_4, atrous_5])
    #
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_2')(x)
    #    
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_4')(x)
    #
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_6')(x)
    #
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_8')(x)

    predictions = Conv2DTranspose(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal', name='Deconv_9')(x)
    
    # Add softmax layer if crf is not on top
    if crf:
        predictions  = softmax(predictions)

    model = Model(inputs=inputs, outputs=predictions)
    
    return model


def SD_CNN(input_shape=(640,960,3), num_classes=20, crf=False):
    inputs = Input(shape=input_shape)
    #
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_1')(inputs)
    x = Conv2D(32, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_1')(x)
    #
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_3')(x)
    x = Conv2D(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_2')(x)
    #
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_5')(x)
    x = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_3')(x)
    #
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal', name='Conv_7')(x)
    x = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal', name='Conv_8')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_4')(x)
    #
    atrous_1 = Conv2D(256, (3,3), dilation_rate=(3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_1')(x)
    atrous_1 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_2')(atrous_1)
    atrous_1 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_3')(atrous_1)
    #
    atrous_2 = Conv2D(256, (3,3), dilation_rate=(6,6), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_1')(x)
    atrous_2 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_2')(atrous_2)
    atrous_2 = Conv2D(128, (1,1), activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_3')(atrous_2)
    #   
    atrous_3 = Conv2D(256, (3,3), dilation_rate=(9,9), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_1')(x)
    atrous_3 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_2')(atrous_3)
    atrous_3 = Conv2D(128, (1,1), padding='valid',  activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_3')(atrous_3)
    #
    atrous_4 = Conv2D(256, (3,3), dilation_rate=(12,12), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_1')(x)
    atrous_4 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_2')(atrous_4)
    atrous_4 = Conv2D(128, (1,1), padding='valid',  activation='selu', kernel_initializer='lecun_normal', name='Atrous_4_3')(atrous_4)
    #
    atrous_5 = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_1')(x)
    atrous_5 = Conv2D(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_2')(atrous_5)
    atrous_5 = Conv2D(128, (1,1), padding='valid', activation='selu', kernel_initializer='lecun_normal', name='Atrous_5_3')(atrous_5)
    x = Add(name='Fusion')([atrous_1, atrous_2, atrous_3, atrous_4, atrous_5])
    #
    x = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_1')(x)
    x = Conv2DTranspose(128, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_2')(x)
    #    
    x = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_3')(x)
    x = Conv2DTranspose(128, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_4')(x)
    #   x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_5')(x)
    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_6')(x)
    #    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal')(x)
    #
    x = Conv2DTranspose(128, (3,3), strides=(2, 2), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_7')(x)
    x = Conv2DTranspose(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_8')(x)
    #
    predictions = Conv2DTranspose(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal', name='Deconv_9')(x)
    
    # Add softmax layer if crf is not on top
    if crf:
        predictions  = softmax(predictions)

    model = Model(inputs=inputs, outputs=predictions)
   
    return model


def test_CNN(input_shape=(640,960,3), num_classes=20, crf=False, batch_size=2):
    #input_shape = (patchSize, patchSize, channels)        
    inputs = Input(shape=input_shape)

    #
    x = Conv2D(64, (5,5), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_1')(inputs)
    x = Conv2D(64, (5,5), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_2')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_1')(x)
    #
    x = Conv2D(128, (5,5), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_3')(x)
    x = Conv2D(128, (5,5), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_4')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_2')(x)
    #
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_5')(x)
    x = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Conv_6')(x)
    x = MaxPooling2D(pool_size=(2, 2), name='Pool_3')(x)
    #
    atrous_1 = Conv2D(256, (3,3), dilation_rate=(6,6), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_1')(x)
    atrous_1 = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_1_2')(atrous_1)
    atrous_1 = Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name='Atrous_1_3')(atrous_1)
    #
    atrous_2 = Conv2D(256, (3,3), dilation_rate=(12,12), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_1')(x)
    atrous_2 = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_2_2')(atrous_2)
    atrous_2 = Conv2D(256, (3,3), activation='selu', kernel_initializer='lecun_normal', padding='same', name='Atrous_2_3')(atrous_2)
    #   
    atrous_3 = Conv2D(256, (3,3), dilation_rate=(18,18), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_1')(x)
    atrous_3 = Conv2D(256, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_2')(atrous_3)
    atrous_3 = Conv2D(256, (3,3), padding='same',  activation='selu', kernel_initializer='lecun_normal', name='Atrous_3_3')(atrous_3)
    #
    """
    print(type(x))
    Average = GlobalAveragePooling2D()(x)
    mat = K.zeros(shape=(batch_size, atrous_3.shape[1],atrous_3.shape[2],                   atrous_3.shape[3]))
    mat = K.permute_dimensions(mat,(2,0,1,3))
    temp = Add(name='Add')([mat, Average])
    Average = K.permute_dimensions(temp, (1,2,0,3))
    #Average = Lambda(lambda Average: Average)(Average)

    print(Average.shape)
    print(type(Average))
    #
    x = Add(name='Fusion')([atrous_1, atrous_2, atrous_3, Average, x])
    #
    """
    print(x.shape)
    print(type(x))

    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(256, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_2')(x)
    x = Conv2DTranspose(256, (3,3),  padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_4')(x)
    #    
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_6')(x)
    x = Conv2DTranspose(128, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_7')(x)
    #
    x = BilinearUpSampling2D(size=(2, 2))(x)
    x = Conv2DTranspose(64, (3,3), padding='same', activation='selu', kernel_initializer='lecun_normal', name='Deconv_8')(x)
    predictions = Conv2DTranspose(num_classes, (1,1), padding='valid', kernel_initializer='lecun_normal', name='Deconv_9')(x)


    model = Model(inputs=inputs, outputs=predictions)
    
    return model
    