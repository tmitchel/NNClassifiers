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
parser.add_argument('--njet', '-N', action='store_true',
                    dest='njet', default=False,
                    help='run on DY + n-jets'
                    )
args = parser.parse_args()
input_length = len(args.vars)

def getWeight(xs, fname):
  from ROOT import TFile
  lumi = 35900.
  fin = TFile('input_files/'+fname, 'r')
  nevnts = fin.Get('nevents').GetBinContent(2)
  return (xs*lumi)/nevnts

cross_sections = {
  0: 1.42383,
  1: 0.45846,
  2: 0.46762,
  3: 0.48084,
  4: 0.39415,
  'VBF125': getWeight(3.782*0.0627, 'VBFHtoTauTau125_svFit_MELA.root'),
}

import h5py
import pandas
import numpy as np
from pprint import pprint
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

def build_nn(nhid):
  """ Build and return the model with callback functions """

  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  # hidden = Dense(nhid, name = 'hidden2', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
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

  print 'Build complete.'
  return model, [early_stopping, model_checkpoint]

def massage_data(vars, fname, sample_type):
  """ read input h5 file, slice out unwanted data, return DataFrame with variables and one-hot """

  print 'Slicing and dicing...', fname.split('.h5')[0].split('input_files/')[-1]
  ifile = h5py.File(fname, 'r')
  slicer = tuple(vars) + ("numGenJets", "Dbkg_VBF", "njets", "pt_sv", "jeta_1", "jeta_2")
  branches = ifile["tt_tree"][slicer]
  df_roc = pandas.DataFrame(branches, columns=['Dbkg_VBF'])
  df = pandas.DataFrame(branches, columns=slicer)
  ifile.close()

  sig_cuts = (df[vars[0]] > -100) & (df['pt_sv'] > 100) & (df['njets'] >= 2) & (abs(df['jeta_1'] - df['jeta_2']) > 2.5)
  if not args.njet:
    bkg_cuts = sig_cuts & (df['numGenJets'] == 2)

  if 'bkg' in sample_type:
    df = df[bkg_cuts]
    df['isSignal'] = np.zeros(len(df))
    df['weight'] = np.array([cross_sections[i] for i in df['numGenJets']])

    df_roc = df_roc[(df_roc['Dbkg_VBF'] > -100)]
    df_roc['isSignal'] = np.zeros(len(df_roc))
  else:
    df = df[sig_cuts]
    df['isSignal'] = np.ones(len(df))
    df['weight'] = np.ones(len(df))

    df_roc = df_roc[(df_roc['Dbkg_VBF'] > -100)]
    df_roc['isSignal'] = np.ones(len(df_roc))

  df = df.drop(['numGenJets', 'Dbkg_VBF', 'njets', 'pt_sv', 'jeta_1', 'jeta_2'], axis=1)
  return df, df_roc

def final_formatting(data, labels):
  """ split data into testing and validation then scale and return collections """
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split

  data_train_val, data_test, label_train_val, label_test = train_test_split(data, labels, test_size=0.2, random_state=7)
  scaler = StandardScaler().fit(data_train_val)
  data_train_val = scaler.transform(data_train_val)
  data_test = scaler.transform(data_test)

  return data_train_val, data_test, label_train_val, label_test

def MELA_ROC(sig, bkg):
  """ read h5 file and return info for making a ROC curve from MELA disc. """
  from sklearn.metrics import roc_curve, auc
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:1]
  labels = dataset[:,1]
  fpr, tpr, thresholds = roc_curve(labels, data)
  roc_auc = auc(fpr, tpr)
  return fpr, tpr, thresholds, roc_auc

def build_plots(history, other=None):
  """ do whatever plotting is needed """
  import  matplotlib.pyplot  as plt

  plt.figure(figsize=(15,10))

  # Plot ROC
  label_predict = model.predict(data_test)
  from sklearn.metrics import roc_curve, auc
  fpr, tpr, thresholds = roc_curve(label_test, label_predict)
  roc_auc = auc(fpr, tpr)
  plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
  plt.plot(tpr, fpr, lw=2, color='cyan', label='NN auc = %.3f' % (roc_auc))
  if other != None:
    fpr2, tpr2, thresholds2, roc_auc2 = other
    plt.plot(tpr2, fpr2, lw=2, color='red', label='MELA auc = %.3f' % (roc_auc2))
  plt.xlim([0, 1.0])
  plt.ylim([0, 1.0])
  plt.xlabel('true positive rate')
  plt.ylabel('false positive rate')
  plt.title('receiver operating curve')
  plt.legend(loc="upper left")
  #plt.show()
  plt.savefig('layer1_node{}_NN.pdf'.format(args.nhid))

  # plot loss vs epoch
  ax = plt.subplot(2, 1, 1)
  ax.plot(history.history['loss'], label='loss')
  ax.plot(history.history['val_loss'], label='val_loss')
  ax.legend(loc="upper right")
  ax.set_xlabel('epoch')
  ax.set_ylabel('loss')

  # plot accuracy vs epoch
  ax = plt.subplot(2, 1, 2)
  ax.plot(history.history['acc'], label='acc')
  ax.plot(history.history['val_acc'], label='val_acc')
  ax.legend(loc="upper left")
  ax.set_xlabel('epoch')
  ax.set_ylabel('acc')
  plt.show()

if __name__ == "__main__":
  ## format the data
  sig, mela_sig = massage_data(args.vars, "input_files/VBFHtoTauTau125_svFit_MELA.h5", "sig")
  bkg, mela_bkg = massage_data(args.vars, "input_files/DYJets2_svFit_MELA.h5", "bkg")
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:input_length]
  labels = dataset[:,input_length:]
  data_train_val, data_test, label_train_val, label_test = final_formatting(data, labels)
  label_train_val, weights = label_train_val[:,0], label_train_val[:,1]
  label_test = label_test[:,0]

  ## build NN
  model, callbacks = build_nn(args.nhid)
  if args.verbose:
    model.summary()

  ## train the NN
  history = model.fit(data_train_val,
                    label_train_val,
                    epochs=5000,
                    batch_size=1024,
                    verbose=args.verbose, # switch to 1 for more verbosity
                    callbacks=callbacks,
                    validation_split=0.25,
                    sample_weight=weights
  )

  build_plots(history, MELA_ROC(mela_sig, mela_bkg))
