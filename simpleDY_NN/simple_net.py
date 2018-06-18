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
args.vars.append('njets')

import os
import h5py
import pandas
import numpy as np
from pprint import pprint
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

input_length = len(args.vars)
def build_nn(nhid):
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  hidden = Dense(nhid, name = 'hidden2', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

  # early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)

  # model checkpoint callback
  model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss',
                     verbose=0, save_best_only=True,
                     save_weights_only=False, mode='auto',
                     period=1)

  return model, [early_stopping, model_checkpoint]

def massage_data(vars, fname, sample_type):
  ifile = h5py.File(fname, 'r')
  slicer = tuple(vars)
  branches = ifile["tt_tree"][slicer]
  test = pandas.DataFrame(branches, columns=vars)
  ifile.close()
  ## still trying to figure out how to slice this with arbitrary number of variables
  test = test[(test[vars[0]] > -100) & (test[vars[1]] > -100)]

  if 'bkg' in sample_type:
    test['isSignal'] = np.zeros(len(test))
  else:
    test['isSignal'] = np.ones(len(test))
  print test.values
  return test

def final_formatting(data, labels):
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split

  data_train_val, data_test, label_train_val, label_test = train_test_split(data, labels, test_size=0.2, random_state=7)
  scaler = StandardScaler().fit(data_train_val)
  data_train_val = scaler.transform(data_train_val)
  data_test = scaler.transform(data_test)

  return data_train_val, data_test, label_train_val, label_test

def build_plots(history):
  import  matplotlib.pyplot  as plt
  # plot loss vs epoch
  plt.figure(figsize=(15,10))
  ax = plt.subplot(2, 2, 1)
  ax.plot(history.history['loss'], label='loss')
  ax.plot(history.history['val_loss'], label='val_loss')
  ax.legend(loc="upper right")
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')

  # plot accuracy vs epoch
  ax = plt.subplot(2, 2, 2)
  ax.plot(history.history['acc'], label='acc')
  ax.plot(history.history['val_acc'], label='val_acc')
  ax.legend(loc="upper left")
  ax.set_xlabel('epoch')
  ax.set_ylabel('acc')

  # Plot ROC
  label_predict = model.predict(data_test)
  from sklearn.metrics import roc_curve, auc
  fpr, tpr, thresholds = roc_curve(label_test, label_predict)
  roc_auc = auc(fpr, tpr)
  ax = plt.subplot(2, 2, 3)
  ax.plot(tpr, fpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
  ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
  ax.set_xlim([0, 1.0])
  ax.set_ylim([0, 1.0])
  ax.set_xlabel('true positive rate')
  ax.set_ylabel('false positive rate')
  ax.set_title('receiver operating curve')
  ax.legend(loc="lower right")
  #plt.show()
  plt.savefig('layer2_node{}_NN.pdf'.format(args.nhid))

if __name__ == "__main__":
  ## build NN
  model, callbacks = build_nn(args.nhid)
  if args.verbose:
    model.summary()

  ## format the data
  sig = massage_data(args.vars, "input_files/VBFHtoTauTau125_svFit_MELA.h5", "sig")
  bkg = massage_data(args.vars, "input_files/DY.h5", "bkg")
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:input_length]
  labels = dataset[:,input_length]
  data_train_val, data_test, label_train_val, label_test = final_formatting(data, labels)

  ## train the NN
  history = model.fit(data_train_val,
                    label_train_val,
                    epochs=1000,
                    batch_size=1024,
                    verbose=1, # switch to 1 for more verbosity
                    callbacks=callbacks,
                    validation_split=0.25)

  build_plots(history)
