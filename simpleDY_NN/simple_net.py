#!/usr/bin/env python

from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'

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
parser.add_argument('--vars', '-v', nargs='+', action='store',
                    dest='vars', default=['Q2V1', 'Q2V2'],
                    help='variables to input to network'
                    )
args = parser.parse_args()

import os
import h5py
import pandas
import numpy as np
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_nn(nhid):
  input_length = len(args.vars)
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
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

def massage_data(vars, fname, sample_type):
  ifile = h5py.File(fname, 'r')
  branches = ifile["tt_tree"][()]
  test = pandas.DataFrame(branches, columns=vars)
  test = test[(test[vars[0]] > -100) & (test[vars[1]] > -100)]
  if 'bkg' in sample_type:
    test['isSignal'] = np.zeros(len(test))
  else:
    test['isSignal'] = np.ones(len(test))
  return test

if __name__ == "__main__":
  model = build_nn(args.nhid)
  # data_train, label_train = massage_data(args.vars, "input_files/VV_svFit_MELA.h5")
  sig = massage_data(args.vars, "input_files/ggHtoTauTau125_svFit_MELA.h5", "sig")
  bkg = massage_data(args.vars, "input_files/DYJets_svFit_MELA.h5", "bkg")
  print sig, bkg
  if args.verbose:
    model.summary()
