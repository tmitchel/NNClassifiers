from os import environ, path, mkdir, listdir
environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import ROOT
import uproot
import pandas as pd
from glob import glob
from array import array
from pprint import pprint
from keras.models import load_model
from collections import defaultdict

def getGuess(df, index):
    try:
        prob_sig = df.loc[index, 'prob_sig']
    except:
        prob_sig = -999
    return prob_sig


def build_filelist(input_dir):
    files = [
        ifile for ifile in glob('{}/*/merged/*.root'.format(input_dir))
    ]

    data = {}
    for fname in files:
        ifile = uproot.open(fname)
        for ikey in ifile.keys():
            if not '_tree' in ikey:
                continue

            if 'SYST_' in fname:
                keyname = fname.split('SYST_')[-1].split('/')[0]
                data.setdefault(keyname, [])
                data[keyname].append(fname)
            else:
                data.setdefault('nominal', [])
                data['nominal'].append(fname)
    pprint(data)
    return data


def main(args):
    if 'mutau' in args.input_dir or 'mtau' in args.input_dir:
        tree_prefix = 'mt_tree'
    elif 'etau' in args.input_dir:
        tree_prefix = 'et_tree'
    else:
        raise Exception(
            'Input files must have MUTAU or ETAU in the provided path. You gave {}, ya goober.'.format(args.input_dir))

    model = load_model('models/{}.hdf5'.format(args.model))
    all_data = pd.HDFStore(args.input_name)

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    filelist = build_filelist(args.input_dir)
    print 'Files to process...'
    pprint(dict(filelist))
    for syst, ifiles in filelist.iteritems():
        if not path.exists('{}/{}'.format(args.output_dir, syst)):
          mkdir('{}/{}'.format(args.output_dir, syst))
        
        for ifile in ifiles:

            ## now let's try and get this into the root file
            root_file = ROOT.TFile(ifile, 'READ')

            # create output file
            fname = ifile.replace('.root', '').split('/')[-1]
            print 'Processing file: {}'.format(fname)
            fout = ROOT.TFile('{}/{}/{}.root'.format(args.output_dir, syst, fname), 'recreate')  ## make new file for output
            fout.cd()
            nevents = root_file.Get('nevents').Clone()
            nevents.Write()
        
            # load the correct tree
            data = all_data[syst]

            ## get dataframe for this sample
            sample = data[(data['sample_names'] == fname) & (data['lepton'] == tree_prefix[:2])]

            ## drop all variables not going into the network
            to_classify = sample[[
                'm_sv', 'mjj', 'higgs_pT', 'Q2V1', 'Q2V2',
                'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar'
            ]]

            ## do the classification
            guesses = model.predict(to_classify.values, verbose=False)
            out = sample.copy()
            out['prob_sig'] = guesses
            out.set_index('idx', inplace=True)

            itree = root_file.Get(tree_prefix)
            ntree = itree.CloneTree(-1, 'fast')

            NN_sig = array('f', [0.])
            disc_branch_sig = ntree.Branch('NN_disc', NN_sig, 'NN_disc/F')
            evt_index = 0
            for _ in itree:
                if evt_index % 100000 == 0:
                    print 'Processing: {}% completed'.format((evt_index*100)/ntree.GetEntries())

                NN_sig[0]= getGuess(out, evt_index)

                evt_index += 1
                fout.cd()
                disc_branch_sig.Fill()
            fout.cd()
            ntree.Write()
        root_file.Close()
        fout.Close()

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of model to use')
    parser.add_argument('--input', '-i', action='store', dest='input_name', default='test', help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--out', '-o', action='store', dest='output_dir', default='output_files/example')

    main(parser.parse_args())
