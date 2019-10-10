from os import environ, path, mkdir, listdir
environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import ROOT
import numpy
import uproot
import pandas as pd
from glob import glob
from array import array
from pprint import pprint
from keras.models import load_model
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

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
            fname = ifile.replace('.root', '').split('/')[-1]
            print 'Processing file: {}'.format(fname)

            open_file = uproot.open(ifile)
            nevents = open_file['nevents']
            oldtree = open_file[tree_prefix].arrays(['*'])
            treedict = {ikey: oldtree[ikey].dtype for ikey in oldtree.keys()}
            treedict['NN_disc'] = numpy.float64

            # load the correct tree
            scaler_info = all_data['scaler']
            scaler_info = scaler_info.drop('isSM', axis=0)

            ## drop all variables not going into the network
            to_classify = open_file[tree_prefix].arrays(scaler_info.index.values, outputtype=pd.DataFrame)

            scaler = StandardScaler()
            scaler.mean_ = scaler_info['mean'].values.reshape(1, -1)
            scaler.scale_ = scaler_info['scale'].values.reshape(1, -1)

            # scale correctly
            scaled = pd.DataFrame(
                scaler.transform(to_classify.values),
                columns=to_classify.columns.values, dtype='float64')

            ## do the classification
            guesses = model.predict(scaled.values, verbose=True)

            with uproot.recreate('{}/{}/{}.root'.format(args.output_dir, syst, fname)) as f:
                f[tree_prefix] = uproot.newtree(treedict)
                oldtree['NN_disc'] = guesses.reshape(len(guesses))
                f[tree_prefix].extend(oldtree)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of model to use')
    parser.add_argument('--input', '-i', action='store', dest='input_name', default='test', help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default='input_files/etau_stable_Oct24', help='name of ROOT input directory')
    parser.add_argument('--out', '-o', action='store', dest='output_dir', default='output_files/example')

    main(parser.parse_args())
