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
    guess = df[(df['idx'] == index)]['guess'].values
    if len(guess) == 1:
        return guess[0]
    elif len(guess) == 0:
        return -999
    else:
        print 'Woah Woah Woah. That definitely shouldn\'t happen'

def main(args):
    model = load_model('models/{}.hdf5'.format(args.model))
    data = pd.HDFStore(args.input_name)['df']
    args.output_dir = 'output_files/' + args.model

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    file_names = [ifile for ifile in glob('{}/*.root'.format(args.input_dir))]
    if args.treename == 'mutau_tree':
        channel = 'mt'
    elif args.treename == 'etau_tree':
        channel = 'et'
    elif args.treename == 'tautau_tree':
        channel = 'tt'
    else:
        print 'Hey. Bad channel. No. Try again.'
        sys.exit(1)


    for ifile in file_names:
        fname = ifile.split('/')[-1].split('.root')[0]
        print 'Processing file: {}'.format(fname)

        ## get dataframe for this sample
        sample = data[(data['sample_names'] == fname) & (data['lepton'] == channel)]

        keep = ['Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar']
        if 'add-mjj-hpt' in args.model:
            keep = ['mjj', 'higgs_pT'] + keep
        elif 'add-mjj' in args.model:
            keep = ['mjj'] + keep
        elif 'add-hpt' in args.model:
            keep = ['higgs_pT'] + keep

        ## drop all variables not going into the network
        to_classify = sample[keep]

        ## do the classification
        guesses = model.predict(to_classify.values, verbose=False)
        sample['guess'] = guesses

        ## now let's try and get this into the root file
        root_file = rt.TFile(ifile, 'READ')
        itree = root_file.Get(args.treename)

        oname = ifile.split('/')[-1].split('.root')[0]
        fout = rt.TFile('{}/{}.root'.format(args.output_dir, oname), 'recreate')  ## make new file for output
        fout.cd()
        nevents = root_file.Get('nevents').Clone()
        nevents.Write()
        # ntree = itree.CloneTree(-1, 'fast')
        ntree = itree.CopyTree("")

        adiscs = array('f', [0.])
        disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')
        evt_index = 0
        for event in itree:
            if evt_index % 100000 == 0:
                print 'Processing: {}% completed'.format((evt_index*100)/ntree.GetEntries())

            adiscs[0] = getGuess(sample, evt_index)
            evt_index += 1
            fout.cd()
            disc_branch.Fill()
        root_file.Close()
        fout.cd()
        ntree.Write()
        fout.Close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--treename', '-t', action='store', dest='treename', default='etau_tree', help='name of input tree')
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of model to use')
    parser.add_argument('--input', '-i', action='store', dest='input_name', default='test', help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--output-dir', '-o', action='store', dest='output_dir', default='output_files', help='name of directory for output')

    main(parser.parse_args())
