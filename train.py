#!/usr/bin/python
import os
from lib.buildModels import bilinear_CNN, SD_CNN, CRFRNN, test_CNN
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from lib.weighted_categorical_crossentropy import weighted_loss
import argparse
import sys
import time
from collections import namedtuple
from keras.optimizers import Adam, SGD
import keras.backend as K
from tensorflow.python import debug as tf_debug
from segDataGenerator import SegDataGenerator
import numpy as np

# Debug commands
if '--debug' in sys.argv:
    sess = K.get_session()
    sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    K.set_session(sess)

NUM_CLASSES = 19
IMG_ROWS = 640
IMG_COLS = 960

# Median Frequency Alpha Coefficients
coefficients = {
        0: 0.0237995754847,
        1: 0.144286494916,
        2: 0.038448897913,
        3: 1.33901803472,
        4: 1.0,
        5: 0.715098627127,
        6: 4.20827446939,
        7: 1.58754122255,
        8: 0.0551054437019,
        9: 0.757994265912,
        10: 0.218245600783,
        11: 0.721125616748,
        12: 6.51048559366,
        13: 0.125434198729,
        14: 3.27995580458,
        15: 3.72813940546,
        16: 3.76817843552,
        17: 8.90686657342,
        18: 2.12162414027,
        19: 0.}

coefficients = [key for index, key in coefficients.iteritems()]

def select_network(args):
    if (args.model is not None) or (args.weights is not None):
        if os.path.exists(args.model) and os.path.exists(args.weights):
            return load_model(args.model, weights=args.weights)

    if args.crf:
        return CRFRNN(args.network)
    else:
        if args.network.lower() == 'bdcnn':
            return bilinear_CNN(input_shape=(IMG_ROWS, IMG_COLS, 3))
        elif args.network.lower() == 'sdcnn':
            return SD_CNN(input_shape=(IMG_ROWS, IMG_COLS, 3))
        elif args.network.lower() == 'testcnn':
            return test_CNN()

def save_model_params(model, model_name, crf=False):
    if not crf:
        crf = 'NO_CRF'
    model_json = model.to_json()
    with open(os.path.join(os.getcwd(), 'model_{}_{}_params.json'.format(model_name, crf)), 'w') as jsonFile:
        jsonFile.write(model_json)
    model.save_weights(os.path.join(os.getcwd(), 'model_{}_{}_weights.h5'.format(model_name, crf)))

def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-n', '--network', help='choose between \'sdcnn\' and \'bdcnn\' networks', type=str)
    parser.add_argument('-trp', '--trainpath',  help='Absolute path of the training set', type=str)
    parser.add_argument('-vdp', '--validationpath', help='Absolute path of the validation set', type=str)
    parser.add_argument('-tsp', '--testpath', help='Absolute path of the test set', type=str)
    parser.add_argument('-bs', '--batchsize', default=1, help='Specify the number of batches', type=int)
    parser.add_argument('-crf', action='store_true', help='Flag to train with CRF module')
    parser.add_argument('-w', '--weights', nargs='?', default=None, const=None, help='The absolute path of the weights', type=str)
    parser.add_argument('-m', '--model', nargs='?', default=None, const=None, help='The absolute path of the model in json format', type=str)
    parser.add_argument('-e', '--epochs', default=None, const=None, help='Specify the number of epochs to train', type=int)

    return parser.parse_args()

def check_paths(args):
    """Check if the path exists
    """
    if os.path.exists(args.trainpath) and os.path.exists(args.validationpath) \
        and os.path.exists(args.testpath):
        return args.trainpath, args.validationpath, args.testpath
    raise Exception('File paths do not exist {}\n {}\n {}'.format(args.trainpath, args.validationpath, args.testpath))
    sys.exit(-1)


def check_args(args):
    """checks and returns the arguments in a namedtuple structure
    """

    params = namedtuple('Parameters', 'network weights model batch crf')
    model_params = params(args.network, None, None, None, False)

    # BatchSize is always 1 to append the crf module
    if args.crf:
        model_params = model_params._replace(batch=1)
        model_params = model_params._replace(crf=True)
    else:
        model_params = model_params._replace(batch=args.batchsize)

    # Load model Parameters passed
    if not args.weights and not args.model:
        model_params = model_params._replace(weights = args.weights)
        model_params = model_params._replace(model = args.model)

    return model_params

def get_file_len(file_path):
    counter = 0
    with open(file_path, 'r') as f:
        for line in f.readlines():
            counter += 1
    return counter

def main(args):

    # Check if paths are in place
    trainFolder, valFolder, testFolder = check_paths(args)

    # Check the remaining model arguments
    parameters = check_args(args)

    # Early stopping to avoid overfitting.
    earlyStopping = EarlyStopping(monitor='val_loss', patience=12)

    # Logger callback for learning curves
    csv_logger = CSVLogger(os.path.join(os.getcwd(), 'train_log.csv'), append=True, separator=',')

    # Checkpoint to save the weights with the best validation accuracy.
    checkPoint = ModelCheckpoint(
            os.path.join(
                os.getcwd(),
                'checkPoint_Weights_{}_.h5'.format(args.network)),
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='min')

    plateauCallback = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.5,
                                        patience=5,
                                        min_lr=0.00005,
                                        verbose=1,
                                        cooldown=3)

    data_gen = SegDataGenerator(
            crop_mode='random',
            crop_size=(IMG_ROWS, IMG_COLS),
            horizontal_flip=True,
            featurewise_center=True,
            featurewise_std_normalization=True)

    batch_size = 1
    steps_per_epoch = int(np.ceil(get_file_len(trainFolder) / float(batch_size)))
    print(steps_per_epoch)

    model = select_network(parameters)
    model.summary()

    model.compile(loss=weighted_loss(NUM_CLASSES+1, coefficients),
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.001),
                  metrics=['accuracy'])

    print(parameters.batch)
    start_time = time.time()
    model.fit_generator(
            generator=data_gen.flow_from_directory(
                trainFolder,
                data_dir='/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/Cityscapes_dataset/leftImg8bit/train/',
                data_suffix='.png',
                label_dir='/media/dimitris/TOSHIBA EXT/UTH/Thesis/Cityscapes_dataset/gtFine/train/',
                loss_shape=(IMG_ROWS, IMG_COLS, NUM_CLASSES+1),
                label_suffix='.png',
                classes=NUM_CLASSES),

            steps_per_epoch=steps_per_epoch,
            epochs=args.epochs,
            workers=4,
            verbose=1,
            callbacks=[earlyStopping, plateauCallback, checkPoint, csv_logger])

    #data_gen.computeTestClasses()
    print("--- %s seconds ---" % (time.time() - start_time))
    save_model_params(model, args.network, args.crf)


if __name__ == '__main__':
    args = argument_parser()
    main(args)
