#!/usr/bin/env python

from argparse import ArgumentParser
parser = ArgumentParser(
  description="Run neural network to separate Z->TT + 2 jets from VBF"
  )
parser.add_argument('--verbose', action='store_true',
                    dest='verbose', default=False,
                    help='run in verbose mode'
                    )
parser.add_argument('--input', '-i', action='store',
                    dest='input', default=None, type=str,
                    help='File with events to input to NN'
                    )
parser.add_argument('--load_json', '-l', action='store',
                    dest='load_json', default='model_store.json',
                    help='name of json file to load model parameters'
                    )
args = parser.parse_args()

import os
import sys
import h5py
import json
import pandas
from os import environ
environ['KERAS_BACKEND'] = 'tensorflow'  ## on Wisc machine, must be before Keras import
from keras.models import Model
from keras.layers import Input, Activation, Dense

## this function needs serious refactoring
def putInTree(fname, discs):
  """Function to write a new file copying old TTree and adding NN discriminant"""
  from ROOT import TFile
  from array import array
  fin = TFile('input_files/'+fname+'.root', 'update')
  itree = fin.Get('tt_tree')
  nentries = itree.GetEntries()
  fout = TFile('output_files/'+fname+'_NN.root', 'recreate')  ## make new file for output
  fout.cd()
  ntree = itree.CloneTree(-1, 'fast')  ## copy all branches from old tree
  adiscs = array('f', [0.])
  disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')  ## make a new branch to store the disc
  j = 0
  for i in range(nentries):
    itree.GetEntry(i)

    if i % 100000 == 0:
      print '{} events of out {} have been processed'.format(i, nentries)

    ## verbose way to do selection, but this step takes a long time so hopefully this will filter out bad events quicker
    if itree.GetLeaf('Q2V1').GetValue() == -100:
      adiscs[0] = -1
    elif itree.GetLeaf('pt_sv').GetValue() < 100:
      adiscs[0] = -1
    elif 'VBF' not in fname and itree.GetLeaf('numGenJets').GetValue() != 2 and not njet:
      adiscs[0] = -1
    # elif abs(itree.GetLeaf('jeta_1').GetValue() - itree.GetLeaf('jeta_2').GetValue()) < 2.5:
    elif itree.GetLeaf('mjj').GetValue() < 300:
      adiscs[0] = -1
    elif itree.GetLeaf('againstElectronVLooseMVA6_1').GetValue() < 0.5 or itree.GetLeaf('againstElectronVLooseMVA6_2').GetValue() < 0.5:
      adiscs[0] = -1
    elif itree.GetLeaf('againstMuonLoose3_1').GetValue() < 0.5 or itree.GetLeaf('againstMuonLoose3_2').GetValue() < 0.5:
      adiscs[0] = -1
    elif itree.GetLeaf('byTightIsolationMVArun2v1DBoldDMwLT_1').GetValue() < 0.5 or itree.GetLeaf('byTightIsolationMVArun2v1DBoldDMwLT_2').GetValue() < 0.5:
      adiscs[0] = -1
    elif itree.GetLeaf('extraelec_veto').GetValue() > 0.5 or itree.GetLeaf('extramuon_veto').GetValue() > 0.5:
      adiscs[0] = -1
    elif itree.GetLeaf('byLooseIsolationMVArun2v1DBoldDMwLT_1').GetValue() < 0.5 and itree.GetLeaf('byLooseIsolationMVArun2v1DBoldDMwLT_2').GetValue() < 0.5:
      adiscs[0] = -1
    else:  ## passes event selection
      adiscs[0] = discs[j][0]
      j += 1

    fout.cd()
    disc_branch.Fill()
  fin.Close()
  fout.cd()
  ntree.Write()
  print '\nin tree'

def build_network(model_name):
  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)
  model.load_weights(model_name)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def create_dataframe():
 ## begin section to run a trained NN on all events in a file
  print 'Loading data...'
  ifile = h5py.File('input_files/'+args.input+'.h5', 'r')

  selection_vars = ['Dbkg_VBF', "numGenJets", "njets", "pt_sv", "jeta_1", "jeta_2", "againstElectronVLooseMVA6_1", "againstElectronVLooseMVA6_2", \
    "againstMuonLoose3_1", "againstMuonLoose3_2", "byTightIsolationMVArun2v1DBoldDMwLT_2", "byTightIsolationMVArun2v1DBoldDMwLT_1", "extraelec_veto", "extramuon_veto",\
    "byLooseIsolationMVArun2v1DBoldDMwLT_2", "byLooseIsolationMVArun2v1DBoldDMwLT_1", "mjj"]

  slicer = tuple(variables) + tuple(selection_vars) ## add event selection variables
  branches = ifile["tt_tree"][slicer]
  df = pandas.DataFrame(branches, columns=slicer)
  ifile.close()

  ## define event selection
  cuts = (df[variables[0]] > -100) & (df['pt_sv'] > 100) & (df['mjj'] > 300) & (df['againstElectronVLooseMVA6_1'] > 0.5) \
    & (df['againstElectronVLooseMVA6_2'] > 0.5) & (df['againstMuonLoose3_1'] > 0.5) & (df['againstMuonLoose3_2'] > 0.5) & (df['byTightIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) \
    & (df['byTightIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) & (df['extraelec_veto'] < 0.5) & (df['extramuon_veto'] < 0.5) \
    & ( (df['byLooseIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) | (df['byLooseIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) )

  if not njet and 'VBF' not in args.input:
    cuts = cuts & (df['numGenJets'] == 2)

  df = df[cuts]  ## apply event selection
  df = df.drop(selection_vars, axis=1)  ## remove unneeded branches from DataFrame
  return df

def normalize(df):
  """Take a pandas DataFrame and normalize the variables"""
  from sklearn.preprocessing import StandardScaler
  from sklearn.model_selection import train_test_split
  scaler = StandardScaler().fit(df)
  data_scaled = scaler.transform(df)
  return data_scaled

if __name__ == "__main__":

  with open(args.load_json, 'r') as fname:
    params = json.load(fname)

  model_name = params['model_name']
  variables  = params['variables']
  njet       = params['njet']
  nhid       = params['nhidden']
  input_length = len(variables)

 
  model_name = 'models/' + model_name + '.hdf5'
  ## load the NN weights
  if not os.path.exists(model_name):
    print "Can't find trained model: {}".format(model_name)
    print "Please run train_network.py -m {}  to train a model named {}".format(model_name, model_name)
    sys.exit(0)

  ## build the network
  model = build_network(model_name)

  ## load data and do event selection
  df = create_dataframe()

  ## normalize the data
  data = normalize(df)

  ## run the NN and make prediction for all events
  print 'Making predictions...'
  predict = model.predict(data, verbose=args.verbose)

  ## put NN discriminant into TTree
  print 'Filling the tree...'
  putInTree(args.input, predict)      
  print 'Done.'