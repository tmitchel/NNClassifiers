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
  fin = TFile(fname, 'update')
  itree = fin.Get('etau_tree')
  nentries = itree.GetEntries()
  oname = fname.split('/')[-1].split('.root')[0]
  fout = TFile('output_files/'+oname+'_NN.root', 'recreate')  ## make new file for output
  fout.cd()
  nevents = fin.Get("nevents").Clone()
  nevents.Write()
  ntree = itree.CloneTree(-1, 'fast')  ## copy all branches from old tree
  adiscs = array('f', [0.])
  disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')  ## make a new branch to store the disc
  
  for i in range(nentries):
    if i % 100000 == 0:
      print '{} events of out {} have been processed'.format(i, nentries)
      adiscs[0] = discs[i][0]
    fout.cd()
    disc_branch.Fill()

  fin.Close()
  fout.cd()
  ntree.Write()

def build_network(model_name):
  print 'Building the network...'
  inputs = Input(shape = (input_length,), name = 'input')
  hidden = Dense(nhid, name = 'hidden', kernel_initializer = 'normal', activation = 'sigmoid')(inputs)
  outputs = Dense(1, name = 'output', kernel_initializer = 'normal', activation = 'sigmoid')(hidden)
  model = Model(inputs = inputs, outputs = outputs)
  model.load_weights(model_name)
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def create_dataframe(variables):
 ## begin section to run a trained NN on all events in a file
  print 'Loading data...'
  fname = args.input.split('.root')[0]
  ifile = h5py.File(fname+'.h5', 'r')

  selection_vars = ['jeta_1', 'jphi_1', 'jeta_2', 'jphi_2']
  # remove duplicates
  selection_vars = [var for var in selection_vars if var not in variables]

  # add event selection variables
  slicer = tuple(variables) + tuple(selection_vars)
  branches = ifile["etau_tree"][slicer]
  df = pandas.DataFrame(branches, columns=slicer)
  ifile.close()

  df.insert(loc=0, column='dEtajj', value=abs(df['jeta_1'] - df['jeta_2']))
  # remove unneeded branches from DataFrame
  df = df.drop(selection_vars, axis=1)
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
  input_length = len(variables) + params['n_user_inputs']

 
  model_name = 'models/' + model_name + '.hdf5'
  ## load the NN weights
  if not os.path.exists(model_name):
    print "Can't find trained model: {}".format(model_name)
    print "Please run train_network.py -m {}  to train a model named {}".format(model_name, model_name)
    sys.exit(0)

  ## build the network
  model = build_network(model_name)

  ## load data and do event selection
  df = create_dataframe(variables)

  ## normalize the data
  data = normalize(df)

  ## run the NN and make prediction for all events
  print 'Making predictions...'
  predict = model.predict(data, verbose=args.verbose)

  ## put NN discriminant into TTree
  print 'Filling the tree...'
  putInTree(args.input, predict)      
  print 'Done.'
