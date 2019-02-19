import sys
import pandas as pd
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

## Variables used for selection. These shouldn't be normalized
selection_vars = [
    'njets', 'OS',
    'is_signal', 'cat_vbf', 'nbjets', 
    'mt', 't1_dmf', 't1_dmf_new', 't1_decayMode'
]

## Variables that could be used as NN input. These should be normalized
scaled_vars = [
    'evtwt', 'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1', 'costheta2', 'costhetastar', 'mjj', 'higgs_pT'
]

def loadFile(ifile):
    from root_pandas import read_root

    if 'mutau' in ifile:
        channel = 'mt'
    elif 'etau' in ifile:
        channel = 'et'
    else:
        raise Exception('Input files must have MUTAU, ETAU, or TAUTAU in the provided path. You gave {}, ya goober.'.format(ifile))

    filename = ifile.split('/')[-1]
    print 'Loading input file...', filename

    columns = scaled_vars + selection_vars
    todrop = ['evtwt', 'idx']
    if 'vbf_inc' in ifile:
        columns = columns + ['wt_a1']
        todrop = todrop + ['wt_a1']

    input_df = read_root(ifile, columns=columns) ## read from TTrees into DataFrame
    input_df['idx'] = np.array([i for i in xrange(0, len(input_df))])
    slim_df = input_df[(input_df['njets'] > 1) & (input_df['mjj'] > 300)] ## preselection
    slim_df = slim_df.dropna(axis=0, how='any') ## drop events with a NaN
    selection_df = slim_df[selection_vars] ## get variables needed for selection (so they aren't normalized)
    weights = slim_df['evtwt'] ## get just the weights (they are scaled differently)
    index = slim_df['idx']
    if 'vbf_inc' in ifile:
        weights = weights * slim_df['wt_a1']
    slim_df = slim_df.drop(selection_vars+todrop, axis=1)

    ## add the event label
    if 'vbf' in ifile.lower() or 'ggh' in ifile.lower():
        isSignal = np.ones(len(slim_df))
    else:
        isSignal = np.zeros(len(slim_df))

    ## save the name of the process
    somenames = np.full(len(slim_df), filename.split('.root')[0])

    ## scale event weights between 0 - 1
    weights = MinMaxScaler(feature_range=(1., 2.)).fit_transform(weights.values.reshape(-1,1))
#    weights = MinMaxScaler(feature_range=(0.1, 1)).fit_transform(weights.values.reshape(-1,1))


    ## get lepton channel
    lepton = np.full(len(slim_df), channel)


    return slim_df, selection_df, somenames, lepton, isSignal, weights, index


def main(args):
    input_files = [ifile for ifile in glob('{}/*.root'.format(args.el_input_dir))]
    input_files += [ifile for ifile in glob('{}/*.root'.format(args.mu_input_dir)) if args.mu_input_dir != None]

    ## define collections that will all be merged in the end
    unscaled_data, selection_df = pd.DataFrame(), pd.DataFrame()
    names, leptons, isSignal, weight_df, index = np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for ifile in input_files:
        input_data, selection_data, new_name, lepton, sig, weight, idx = loadFile(ifile)
        unscaled_data = pd.concat([unscaled_data, input_data])  ## add data to the full set
        selection_df = pd.concat([selection_df, selection_data]) ## add selection variables to full set
        names = np.append(names, new_name) ## insert the name of the current sample
        isSignal = np.append(isSignal, sig) ## labels for signal/background
        weight_df = np.append(weight_df, weight) ## weights scaled from 0 - 1
        leptons = np.append(leptons, lepton) ## lepton channel
        index = np.append(index, idx)

    ## normalize the potential training variables
    scaled_data = pd.DataFrame(StandardScaler().fit_transform(unscaled_data.values), columns=unscaled_data.columns.values)

    ## add selection variables
    for column in selection_df.columns:
        scaled_data[column] = selection_df[column].values

    ## add other useful data
    scaled_data['sample_names'] = pd.Series(names)
    scaled_data['lepton'] = pd.Series(leptons)
    scaled_data['isSignal'] = pd.Series(isSignal)
    scaled_data['evtwt'] = pd.Series(weight_df)
    scaled_data['idx'] = pd.Series(index)

    ## save the dataframe for later
    store = pd.HDFStore('datasets/{}.h5'.format(args.output))
    store['df'] = scaled_data

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--el-input', '-e', action='store', dest='el_input_dir', default=None, help='path to etau input files')
    parser.add_argument('--mu-input', '-m', action='store', dest='mu_input_dir', default=None, help='path to mutau input files')
    parser.add_argument('--output', '-o', action='store', dest='output', default='store.h5', help='name of output file')

    main(parser.parse_args())
