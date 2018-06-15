#!/usr/bin/env python

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

import os
import h5py
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from argparse import ArgumentParser

parser = ArgumentParser(description="Run neural network to separate Z->TT + 2 jets from VBF")
parser.add_argument('--verbose', action='store_true',
                    dest='verbose', default=False,
                    help='run in verbose mode'
                    )
parser.add_argument('--nhid', '-n', action='store',
                    dest='nhid', default=5, type=int,
                    help='number of hidden nodes in network'
                    )
parser.add_argument('--rebuild', action='store_true',
                    dest='rebuild', default=False,
                    help='rebuild the nn model'
                    )
args = parser.parse_args()

def build_nn(nhid):
  input_length = 7
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(2, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)

  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)

  # model checkpoint callback
  model_checkpoint = ModelCheckpoint('simple.hdf5', monitor='val_loss',
  verbose=0, save_best_only=True,
  save_weights_only=False, mode='auto',
  period=1)

  return model

if __name__ == "__main__":
  model = build_nn(args.nhid)
  if args.verbose:
    model.summary()
