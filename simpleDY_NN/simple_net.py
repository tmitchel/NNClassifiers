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
parser.add_argument('--addToFile', '-a', action='store_true',
                    dest='addToFile', default=False,
                    help='Add NN discriminant to tree'
                    )
parser.add_argument('--input', '-i', action='store',
                    dest='input', default=None, type=str,
                    help='File with events to input to NN'
                    )
parser.add_argument('--retrain', '-r', action='store_true',
                    dest='retrain', default=False,
                    help='retrain the NN'
                    )
args = parser.parse_args()
input_length = len(args.vars)

def getWeight(xs, fname):
  """Return SF for normalization of a given sample at lumi=35900 fb^-1"""
  from ROOT import TFile
  lumi = 35900.
  fin = TFile('input_files/'+fname, 'r')
  nevnts = fin.Get('nevents').GetBinContent(2)
  return (xs*lumi)/nevnts

VBFweight = getWeight(3.782*0.0627, 'VBFHtoTauTau125_svFit_MELA.root')
cross_sections = {
  0: 1.42383,
  1: 0.45846,
  2: 0.46762,
  3: 0.48084,
  4: 0.39415,
  'VBF125': VBFweight,
}

import os
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
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)

  ## callbacks start empty since they aren't needed when loading pretrained model
  callbacks = []

  ## load the pretrained model if it exists
  if os.path.exists('model_checkpoint.hdf5') and not args.retrain:
    model.load_weights('model_checkpoint.hdf5')
  else:
    # early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    # model checkpoint callback
    model_checkpoint = ModelCheckpoint('model_checkpoint.hdf5', monitor='val_loss',
                       verbose=0, save_best_only=True,
                       save_weights_only=False, mode='auto',
                       period=1)
    callbacks = [early_stopping, model_checkpoint]

  ## compile the model and return it with callbacks
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  print 'Build complete.'
  return model, callbacks

def massage_data(vars, fname, sample_type):
  """ read input h5 file, slice out unwanted data, return DataFrame with variables and one-hot """

  print 'Slicing and dicing...', fname.split('.h5')[0].split('input_files/')[-1]
  ifile = h5py.File(fname, 'r')
  slicer = tuple(vars) + ("numGenJets", "Dbkg_VBF", "njets", "pt_sv", "jeta_1", "jeta_2")  ## add variables for event selection
  branches = ifile["tt_tree"][slicer]
  df_roc = pandas.DataFrame(branches, columns=['Dbkg_VBF'])  ## DataFrame used for MELA ROC curve
  df = pandas.DataFrame(branches, columns=slicer)  ## DataFrame holding NN input
  ifile.close()

  ## define event selection
  sig_cuts = (df[vars[0]] > -100) & (df['pt_sv'] > 100) & (df['njets'] >= 2) & (abs(df['jeta_1'] - df['jeta_2']) > 2.5)
  if not args.njet:
    bkg_cuts = sig_cuts & (df['numGenJets'] == 2)

  if 'bkg' in sample_type:

    ## format background DataFrame for NN input
    df = df[bkg_cuts]
    df['isSignal'] = np.zeros(len(df))  ## put label in DataFrame (bkg=0)

    ## apply physics normalization if DY+Njets
    if args.njet:
      df['weight'] = np.array([cross_sections[i] for i in df['numGenJets']])
    else:
      df['weight'] = np.ones(len(df))

    ## format bkg DataFrame for MELA ROC curve
    df_roc = df_roc[(df_roc['Dbkg_VBF'] > -100)]
    df_roc['isSignal'] = np.zeros(len(df_roc))

  else:

    ## format signal DataFrame for NN input
    df = df[sig_cuts]
    df['isSignal'] = np.ones(len(df))  ## put label in DataFrame (sig=1)

    ## apply physics normalization if DY+Njets
    if args.njet:
      df['weight'] = np.array([cross_sections['VBF125'] for i in range(len(df))])
    else:
      df['weight'] = np.ones(len(df))

    ## format bkg DataFrame for MELA ROC curve
    df_roc = df_roc[(df_roc['Dbkg_VBF'] > -100)]
    df_roc['isSignal'] = np.ones(len(df_roc))

  ## drop event selection branches from NN input
  df = df.drop(['numGenJets', 'Dbkg_VBF', 'njets', 'pt_sv', 'jeta_1', 'jeta_2'], axis=1)
  return df, df_roc

def final_formatting(data, labels):
  """ split data into testing and validation then scale and return collections """
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split

  ## split data into labels and also split into train/test
  data_train_val, data_test, label_train_val, label_test = train_test_split(data, labels, test_size=0.2, random_state=7)

  ## normalize all input variables to improve performance
  scaler = StandardScaler().fit(data_train_val)
  data_train_val = scaler.transform(data_train_val)
  data_test = scaler.transform(data_test)

  ## return the same that train_test_split does, but normalized
  return data_train_val, data_test, label_train_val, label_test

def MELA_ROC(sig, bkg):
  """ read h5 file and return info for making a ROC curve from MELA disc. """
  from sklearn.metrics import roc_curve, auc
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:1]  ## read Dbkg_VBF for all events
  labels = dataset[:,1]  ## read labels for all events
  fpr, tpr, thresholds = roc_curve(labels, data)
  roc_auc = auc(fpr, tpr)  ## calculate Area Under Curve
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

## this function needs serious refactoring
def putInTree(fname, discs):
  """Function to write a new file copying old TTree and adding NN discriminant"""
  from ROOT import TFile, TTree, TBranch
  from array import array
  fin = TFile('input_files/'+fname+'.root', 'update')
  itree = fin.Get('tt_tree')
  nentries = itree.GetEntries()
  fout = TFile('input_files/'+fname+'_NN.root', 'recreate')  ## make new file for output
  fout.cd()
  ntree = itree.CloneTree()  ## copy all branches from old tree
  adiscs = array('f', [0.])
  disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')  ## make a new branch to store the disc
  j = 0
  for i in range(nentries):
    itree.GetEntry(i)

    ## verbose way to do selection, but this step takes a long time so hopefully this will filter out bad events quicker
    if itree.GetLeaf('Q2V1').GetValue() == -100:
      adiscs[0] = -1
    elif itree.GetLeaf('pt_sv').GetValue() < 100:
      adiscs[0] = -1
    elif itree.GetLeaf('njets').GetValue() < 2:
      adiscs[0] = -1
    elif 'VBF' not in fname and itree.GetLeaf('numGenJets').GetValue() != 2:
      adiscs[0] = -1
    elif abs(itree.GetLeaf('jeta_1').GetValue() - itree.GetLeaf('jeta_2').GetValue()) > 2.5:
      adiscs[0] = -1
    else:  ## passes event selection
      adiscs[0] = discs[j][0]
      j += 1

    fout.cd()
    disc_branch.Fill()
  fin.Close()
  fout.cd()
  ntree.Write()
  print 'in tree'

if __name__ == "__main__":
  ## format the data
  sig, mela_sig = massage_data(args.vars, "input_files/VBFHtoTauTau125_svFit_MELA.h5", "sig")
  bkg, mela_bkg = massage_data(args.vars, "input_files/DYJets2_svFit_MELA.h5", "bkg")
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:input_length]  ## get numpy array with all input variables
  labels = dataset[:,input_length:]  ## get numpy array with weights and labels
  data_train_val, data_test, label_train_val, label_test = final_formatting(data, labels) ## split/normalize/randomize data
  label_train_val, weights = label_train_val[:,0], label_train_val[:,1]
  label_test = label_test[:,0]

  ## build new NN or load premade NN
  model, callbacks = build_nn(args.nhid)
  if args.verbose:
    model.summary()

  if len(callbacks) > 0:
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

    if args.verbose:
      build_plots(history, MELA_ROC(mela_sig, mela_bkg)) ## make ROC curve and other plots

  if args.input != None:
    ## begin section to run a trained NN on all events in a file
    ifile = h5py.File('input_files/'+args.input+'.h5', 'r')
    slicer = tuple(args.vars) + ("numGenJets", "njets", "pt_sv", "jeta_1", "jeta_2") ## add event selection variables
    branches = ifile["tt_tree"][slicer]
    df = pandas.DataFrame(branches, columns=slicer)
    ifile.close()

    ## define event selection
    cuts = (df[args.vars[0]] > -100) & (df['pt_sv'] > 100) & (df['njets'] >= 2) & (abs(df['jeta_1'] - df['jeta_2']) > 2.5)
    if not args.njet and 'VBF' not in args.input:
      cuts = cuts & (df['numGenJets'] == 2)

    df = df[cuts]  ## apply event selection
    df = df.drop(['numGenJets', 'njets', 'pt_sv', 'jeta_1', 'jeta_2'], axis=1)  ## remove unneeded branches from DataFrame

    ## normalize the variables
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    scaler = StandardScaler().fit(df)
    data_scaled = scaler.transform(df)

    ## run the NN and make prediction for all events
    predict = model.predict(data_scaled, verbose=args.verbose)

    ## put NN discriminant into TTree
    putInTree(args.input, predict)
