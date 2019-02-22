from __future__ import print_function
import sys
import ROOT as rt
import pandas as pd
from glob import glob
from array import array
from os import environ, path, mkdir
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model
import matplotlib.pyplot as plt

def getGuess(df, index):
    try:
      guess = df.loc[index, 'guess']
    except:
      guess = -999
    return guess

def main(args):
    model_vbf = load_model('models/{}.hdf5'.format(args.model_vbf))
    model_boost = load_model('models/{}.hdf5'.format(args.model_boost))
    data_vbf = pd.HDFStore(args.input_vbf)['df']
    data_boost = pd.HDFStore(args.input_boost)['df']

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    file_names = [ifile for ifile in glob('{}/*.root'.format(args.input_dir))]
    if args.treename == 'mutau_tree':
        channel = 'mt'
    elif args.treename == 'etau_tree':
        channel = 'et'
    else:
        raise Exception('Hey. Bad channel. No. Try again.')


    for ifile in file_names:
        fname = ifile.split('/')[-1].split('.root')[0]
        print 'Processing file: {}'.format(fname)

        ## get dataframe for this sample
        sample_vbf = data_vbf[
            (data_vbf['sample_names'] == fname) & (data_vbf['lepton'] == channel)
        ].copy()
        sample_boost = data_boost[
            (data_boost['sample_names'] == fname) & (data_boost['lepton'] == channel)
        ].copy()

        ## drop all variables not going into the network
        keep_vbf = ['m_sv', 'mjj', 'higgs_pT', 'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar']
        keep_boost = ['higgs_pT', 't1_pt', 'MT_t2MET', 'MT_HiggsMET', 'jmet_dphi']
        to_classify_vbf = sample_vbf[keep_vbf]
        to_classify_boost = sample_boost[keep_boost]

        ## do the classification
        guesses_vbf = model_vbf.predict(to_classify_vbf.values, verbose=False)
        guesses_boost = model_boost.predict(to_classify_boost.values, verbose=False)
        sample_vbf['guess'] = guesses_vbf
        sample_boost['guess'] = guesses_boost
        sample_vbf.set_index('idx', inplace=True)
        sample_boost.set_index('idx', inplace=True)

        ## now let's try and get this into the root file
        root_file = rt.TFile(ifile, 'READ')
        itree = root_file.Get(args.treename)

        oname = ifile.split('/')[-1].split('.root')[0]
        fout = rt.TFile('{}/{}.root'.format(args.output_dir, oname), 'recreate')  ## make new file for output
        fout.cd()
        nevents = root_file.Get('nevents').Clone()
        nevents.Write()
        ntree = itree.CloneTree(-1, 'fast')

        branch_var = array('f', [0.])
        branch_var_vbf = array('f', [0.])
        branch_var_boost = array('f', [0.])
        disc_branch = ntree.Branch('NN_disc', branch_var, 'NN_disc/F')
        disc_branch_vbf = ntree.Branch('NN_disc_vbf', branch_var_vbf, 'NN_disc_vbf/F')
        disc_branch_boost = ntree.Branch('NN_disc_boost', branch_var_boost, 'NN_disc_boost/F')
        nevts = ntree.GetEntries()
        evt_index = 0
        progress = 1
        fraction = nevts / 10
        for _ in itree:
            if evt_index == progress * fraction
                print('{} % complete'.format(progress*10), end='\r', flush=True)
                progress+=1

            branch_var[0] = getGuess(sample_vbf, evt_index)
            branch_var_vbf[0] = getGuess(sample_vbf, evt_index)
            branch_var_boost[0] = getGuess(sample_boost, evt_index)
            evt_index += 1
            fout.cd()
            disc_branch.Fill()
            disc_branch_vbf.Fill()
            disc_branch_boost.Fill()
        root_file.Close()
        fout.cd()
        ntree.Write()
        fout.Close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--treename', '-t', action='store', dest='treename', default='etau_tree', help='name of input tree')
    parser.add_argument('--model-vbf', action='store', dest='model_vbf', default=None, help='name of model to use')
    parser.add_argument('--model-boost', action='store', dest='model_boost', default=None, help='name of model to use')
    parser.add_argument('--input-vbf', action='store', dest='input_vbf', default=None, help='name of input dataset')
    parser.add_argument('--input-boost', action='store', dest='input_boost', default=None, help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--output-dir', '-o', action='store', dest='output_dir', default='output_files', help='name of directory for output')

    main(parser.parse_args())
