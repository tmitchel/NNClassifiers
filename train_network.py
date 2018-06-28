#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser(
  description = "Train two-layer neural network to separate Drell-Yan + jets from VBF Higgs to tau-tau"
  )
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
parser.add_argument('--model_name', '-m', action='store',
                    dest='model_name', default=None,
                    help='name of a trained model'
                    )
parser.add_argument('--save_json', '-s', action='store_false',
                    dest='save_json', default=True,
                    help="don't store NN settings to json"
                    )

args = parser.parse_args()
# input_length = len(args.vars)

import h5py
import pandas
import numpy as np
from os import environ
from pprint import pprint
environ['KERAS_BACKEND'] = 'tensorflow'  ## on Wisc machine, must be before Keras import
from keras.models import Model
from keras.layers import Input, Activation, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

def create_json(model_name):
  import json

  if args.model_name != None:
    fname = args.model_name + '.json'
  else:
    fname = 'model_store.json'

  with open('model_params/'+fname, 'w') as fout:
    json.dump(
      {
        'model_name': model_name,
        'variables': args.vars,
        'nhidden': args.nhid,
        'njet': args.njet
      }, fout
    )

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

def build_nn(nhid):
  """ Build and return the model with callback functions """

  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)

  # early stopping callback
  early_stopping = EarlyStopping(monitor='val_loss', patience=10)

  if args.model_name != None:
    model_name = args.model_name + '.hdf5'
  else:
    if args.njet:
      model_name = 'NN_njet_model.hdf5'
    else:
      model_name = 'NN_2jet_model.hdf5'

  model_name = 'models/' + model_name

  # model checkpoint callback
  model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
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

  selection_vars = ['Dbkg_VBF', "numGenJets", "njets", "pt_sv", "jeta_1", "jeta_2", "againstElectronVLooseMVA6_1", "againstElectronVLooseMVA6_2", \
    "againstMuonLoose3_1", "againstMuonLoose3_2", "byTightIsolationMVArun2v1DBoldDMwLT_2", "byTightIsolationMVArun2v1DBoldDMwLT_1", "extraelec_veto", "extramuon_veto",\
    "byLooseIsolationMVArun2v1DBoldDMwLT_2", "byLooseIsolationMVArun2v1DBoldDMwLT_1", "mjj"]

  selection_vars = [var for var in selection_vars if var not in vars]
  
  slicer = tuple(vars) + tuple(selection_vars)  ## add variables for event selection
  branches = ifile["tt_tree"][slicer]
  df_roc = pandas.DataFrame(branches, columns=['Dbkg_VBF'])  ## DataFrame used for MELA ROC curve
  df = pandas.DataFrame(branches, columns=slicer)  ## DataFrame holding NN input
  ifile.close()

  ## additional variables for selection that must be constructed can be added here
  ## ...

  ## define event selection
  mela_cuts = (df['Dbkg_VBF'] > -100)
  sig_cuts = (df['Q2V1'] > -100)
  evt_cuts = (df['pt_sv'] > 100) & (df['mjj'] > 300) & (df['againstElectronVLooseMVA6_1'] > 0.5) \
    & (df['againstElectronVLooseMVA6_2'] > 0.5) & (df['againstMuonLoose3_1'] > 0.5) & (df['againstMuonLoose3_2'] > 0.5) & (df['byTightIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) \
    & (df['byTightIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) & (df['extraelec_veto'] < 0.5) & (df['extramuon_veto'] < 0.5) \
    & ( (df['byLooseIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) | (df['byLooseIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) )  

  sig_cuts = sig_cuts & evt_cuts
  mela_cuts = mela_cuts & evt_cuts

  if 'bkg' in sample_type:

    ## choose DY + 2-Jets or DY + N-Jets
    bkg_cuts = sig_cuts
    if not args.njet:
      bkg_cuts = bkg_cuts & (df['numGenJets'] == 2)

    ## format background DataFrame for NN input
    df = df[bkg_cuts]
    df['isSignal'] = np.zeros(len(df))  ## put label in DataFrame (bkg=0)

    ## get cross section normalization
    df['weight'] = np.array([cross_sections[i] for i in df['numGenJets']])

    ## format bkg DataFrame for MELA ROC curve
    df_roc = df_roc[bkg_cuts]
    df_roc['isSignal'] = np.zeros(len(df_roc))

  else:

    ## format signal DataFrame for NN input
    df = df[sig_cuts]
    df['isSignal'] = np.ones(len(df))  ## put label in DataFrame (sig=1)

    ## get cross section normalization
    df['weight'] = np.array([cross_sections['VBF125'] for i in range(len(df))])

    ## format bkg DataFrame for MELA ROC curve
    df_roc = df_roc[sig_cuts]
    df_roc['isSignal'] = np.ones(len(df_roc))

  ## additional input variables that must be constructed can be added here
  df.insert(loc=0, column='dEtajj', value=abs(df['jeta_1'] - df['jeta_2']))  ## add to beginning because weight/isSignal must be last

  ## drop event selection branches from NN input
  df = df.drop(selection_vars, axis=1)
  return df, df_roc

def build_plots(history, label_test, other=None):
  """ do whatever plotting is needed """
  import  matplotlib.pyplot  as plt
  from sklearn.metrics import roc_curve, auc

  plt.figure(figsize=(15,10))

  # Plot ROC
  label_predict = model.predict(data_test)
  fpr, tpr, thresholds = roc_curve(label_test[:,0], label_predict[:,0])
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
  if args.njet:
    ext = 'njet'
    plt.savefig('plots/layer1_node{}_njet_NN.pdf'.format(args.nhid))
  else:
    plt.savefig('plots/layer1_node{}_2jet_NN.pdf'.format(args.nhid))

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
  labels = dataset[:,-1]  ## read labels for all events
  fpr, tpr, thresholds = roc_curve(labels, data[:,0])
  roc_auc = auc(fpr, tpr)  ## calculate Area Under Curve
  return fpr, tpr, thresholds, roc_auc

if __name__ == "__main__":
  ## format the data
  sig, mela_sig = massage_data(args.vars, "input_files/VBFHtoTauTau125_svFit_MELA.h5", "sig")
  input_length = sig.shape[1] - 2  ## get shape and remove weight & isSignal
  bkg, mela_bkg = massage_data(args.vars, "input_files/DY.h5", "bkg")
  all_data = pandas.concat([sig, bkg])
  dataset = all_data.values
  data = dataset[:,0:input_length]  ## get numpy array with all input variables
  labels = dataset[:,input_length:]  ## get numpy array with weights and labels
  data_train_val, data_test, label_train_val, label_test = final_formatting(data, labels) ## split/normalize/randomize data
  label_train_val, weights = label_train_val[:,0], label_train_val[:,1]  ## split into labels and weights

  ## if running DY + 2-Jets, set all weights to 1.0
  weights = np.array([1. for i in range(len(label_train_val))])

  ## build the NN
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

  if args.verbose:
    ## produce a ROC curve and other plots
    build_plots(history, label_test, MELA_ROC(mela_sig, mela_bkg))

  if args.save_json:
    if args.model_name != None:
      model_name = args.model_name
    elif args.njet:
      model_name = 'NN_njet_model'
    else:
      model_name = 'NN_2jet_model'

    create_json(model_name)