#!/usr/bin/env python

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, EarlyStopping
