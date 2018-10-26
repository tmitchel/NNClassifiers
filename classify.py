import ROOT as rt
import pandas as pd
from glob import glob
from array import array
from keras.models import load_model

def main(args):
    model = load_model('models/{}.hdf5'.format(args.model))
    data = pd.HDFStore(args.input_name)['df']

    file_names = [ifile for ifile in glob('{}/*.root'.format(args.input_dir))]

    for ifile in file_names:
        print 'Processing file: {}'.format(ifile.split('/')[-1].split('.root')[0])
        ## get dataframe for this sample
        sample = data[(data['sample_names'] == ifile.split('/')[-1])]

        ## drop all variables not going into the network
        to_classify = sample[
            ['Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar']
        ]

        ## do the classification
        guesses = model.predict(to_classify.values, verbose=False)

        ## now let's try and get this into the root file
        root_file = rt.TFile(ifile, 'READ')
        itree = root_file.Get(args.treename)

        oname = ifile.split('/')[-1].split('.root')[0]
        fout = rt.TFile('output_files/{}.root'.format(oname), 'recreate')  ## make new file for output
        fout.cd()
        nevents = root_file.Get('nevents').Clone()
        nevents.Write()
        ntree = itree.CloneTree(-1, 'fast')

        adiscs = array('f', [0.])
        disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')
        evt_index = 0
        guess_index = 0
        for event in itree:
            if evt_index % 100000 == 0:
                print 'Processing: {}% completed'.format((evt_index*100)/ntree.GetEntries())

            if event.njets > 1 and event.mjj > 300 and event.mt < 50:
                adiscs[0] = guesses[guess_index]
                guess_index += 1
            else:
                adiscs[0] = -999

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

    main(parser.parse_args())