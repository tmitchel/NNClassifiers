from os import environ, path, mkdir, listdir
environ['KERAS_BACKEND'] = 'tensorflow'
import sys
import ROOT
import uproot
import pandas as pd
from glob import glob
from array import array
from keras.models import load_model

def getGuess(df, index):
    try:
        prob_sig = df.loc[index, 'prob_sig']
    except:
        prob_sig = -999
    return prob_sig


def build_filelist(input_dir):

    filelist = {}
    for fname in glob('{}/*.root'.format(input_dir)):
        ifile = uproot.open(fname)
        for ikey in ifile.keys():
            if not '_tree' in ikey:
                continue

            keyname = ikey.replace(';1', '')
            filelist[keyname] = fname

    return filelist

def main(args):
    if 'mutau' in args.input_dir:
        tree_prefix = 'mutau_tree'
    elif 'etau' in args.input_dir:
        tree_prefix = 'etau_tree'
    else:
        raise Exception(
            'Input files must have MUTAU or ETAU in the provided path. You gave {}, ya goober.'.format(args.input_dir))

    model = load_model('models/{}.hdf5'.format(args.model))
    all_data = pd.HDFStore(args.input_name)

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    filelist = build_filelist(args.input_dir)
    for ifile, dirs in filelist.iteritems():
        mkdir(args.output_dir)
        ## now let's try and get this into the root file
        root_file = ROOT.TFile(ifile, 'READ')

        # create output file
        fname = ifile.replace('.root', '')
        fout = ROOT.TFile('{}/{}.root'.format(args.output_dir, fname), 'recreate')  ## make new file for output
        fout.cd()
        nevents = root_file.Get('nevents').Clone()
        nevents.Write()
        
        # loop through trees
        for idir in dirs:
            if not 'tree' in idir:
                continue
            print 'Processing file: {}'.format(fname)

            syst = idir.replace('etau_tree_', '')
            syst = syst.replace('mutau_tree_', '')
            data = all_data[syst] # load the correct tree

            ## get dataframe for this sample
            sample = data[(data['sample_names'] == fname)]

            ## drop all variables not going into the network
            to_classify = sample[[
                'm_sv', 'mjj', 'higgs_pT', 'Q2V1', 'Q2V2',
                'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar'
            ]]

            ## do the classification
            guesses = model.predict(to_classify.values, verbose=False)
            out = sample.copy()
            out['prob_sig'] = guesses[:, 0]
            out.set_index('idx', inplace=True)


            itree = root_file.Get(idir)
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
