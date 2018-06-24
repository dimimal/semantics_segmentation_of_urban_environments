#!/usr/bin/python

'''
This script loads a keras model from json and 
outputs the predictions from images 
'''
from __future__ import print_function
from keras.preprocessing import image
import keras
import matplotlib.pyplot as plt
from keras import backend as K
from lib.dataGenerator import DataGenerator 
from lib.utils import computeGlobalIou, plot_confusion_matrix, load_model
from sklearn.metrics import f1_score, classification_report, confusion_matrix, jaccard_similarity_score, accuracy_score
from lib.weighted_categorical_crossentropy import weighted_loss
from lib.bilinearUpsampling import BilinearUpSampling2D
import random
import numpy as np
import os
import itertools
import labels  
from keras.models import Model, Sequential
from collections import namedtuple
import cv2 as cv
import sys
import tensorflow as tf
import time
import sys
import re
import argparse 
import pandas as pd
import csv
from lib.crfasrnn_keras.src.crfrnn_layer import CrfRnnLayer 


NUM_CLASSES         = 20
BATCH_SIZE          = 1
patchSize           = 512
CHANNELS            = 3
img_rows, img_cols  = patchSize, patchSize


def argument_parser():
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('-w', '--weights', nargs='?', default=None, const=None, help='The absolute path of the weights of the model', type=str)
    parser.add_argument('-m', '--model', nargs='?', default=None, const=None, help='The absolute path of the model in json format', type=str)
    parser.add_argument('-file', '--file', nargs='?', default=None, const=None, help='The absolute \
    	path of the folder containing the images for evaluation', type=str)
    parser.add_argument('-im', '--image', nargs='?', default=None, const=None, help='The absolute path of the weights', type=str)
    parser.add_argument('-mf', '--medianFilter', nargs='?', default=None, const=None, help='The size of the median filter kernel', type=int)

    return parser.parse_args()

def function():
	pass

def main(args):
	# Image prediction or evaluation path
	if not args.im:
		mode = 'image'
	else:
		mode = 'evaluation'

	if args.model:
		model = load_model(args.model)
	else: 
		raise Exception('Model not given {}'.format(args.model))


if __name__ == '__main__':
	args = argument_parser()
	main(args)