import sys
import uproot
import numpy as np
import pandas as pd
from glob import glob
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import warnings
warnings.filterwarnings(
    'ignore', category=pd.io.pytables.PerformanceWarning)

# Variables used for selection. These shouldn't be normalized
selection_vars = ['njets', 'OS', 'is_signal']

# Variables that could be used as NN input. These should be normalized
scaled_vars = {
    'vbf': [
        'evtwt', 'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1',
        'costheta2', 'costhetastar', 'mjj', 'higgs_pT', 'm_sv',
    ],
    'boosted': [
        'evtwt', 'higgs_pT', 't1_pt', 'lt_dphi', 'lep_pt',
        'hj_dphi', 'MT_lepMET', 'MT_HiggsMET', 'met', 'mjj'
    ]
}


def loadFile(ifile, open_file, itree, category):
    if 'mutau' in ifile:
        channel = 'mt'
        syst = itree.replace('mutau_', '')
    elif 'etau' in ifile:
        channel = 'et'
        syst = itree.replace('etau_', '')
    else:
        raise Exception(
            'Input files must have MUTAU or ETAU in the provided path. You gave {}, ya goober.'.format(ifile))

    filename = ifile.split('/')[-1]
    print 'Loading input file...', filename, itree

    todrop = ['evtwt', 'idx']
    columns = scaled_vars[category] + selection_vars
    if 'vbf_inc' in ifile:
        columns = columns + ['wt_a1']
        todrop = todrop + ['wt_a1']

    # read from TTrees into DataFrame
    input_df = open_file[itree].pandas.df(columns)
    input_df['idx'] = np.array([i for i in xrange(0, len(input_df))])

    # preselection
    if category == 'vbf':
        slim_df = input_df[
            (input_df['njets'] > 1) & (input_df['mjj'] > 300)
        ]
    elif category == 'boosted':
        slim_df = input_df[
            (input_df['njets'] == 1)
        ]
    else:
        raise Exception('Not a category: {}'.format(category))

    slim_df = slim_df.dropna(axis=0, how='any')  # drop events with a NaN
    slim_df = slim_df.drop_duplicates()  # drop duplicate events

    # combine lepton pT's
    if category == 'boosted':
        if channel == 'mt':
            slim_df['lep_pt'] = slim_df['mu_pt'].copy()
        elif channel == 'et':
            slim_df['lep_pt'] = slim_df['el_pt'].copy()
        slim_df = slim_df.drop(['el_pt', 'mu_pt'], axis=1)

    # get variables needed for selection (so they aren't normalized)
    selection_df = slim_df[selection_vars]

    # get just the weights (they are scaled differently)
    weights = slim_df['evtwt']
    index = slim_df['idx']
    if 'vbf_inc' in ifile:
        weights = weights * slim_df['wt_a1']
    slim_df = slim_df.drop(selection_vars+todrop, axis=1)
    slim_df = slim_df.astype('float64')

    # add the event label
    if 'vbf' in ifile.lower() or 'ggh' in ifile.lower():
        isSignal = np.ones(len(slim_df))
    else:
        isSignal = np.zeros(len(slim_df))

    # scale event weights between 1 - 2
    weights = MinMaxScaler(
        feature_range=(1., 2.)
    ).fit_transform(
        weights.values.reshape(-1, 1)
    )

    return {
        'slim_df': slim_df,
        'selection_df': selection_df,
        'isSignal': isSignal,
        'weights': weights,
        'index': index,
        'somenames': np.full(len(slim_df), filename.split('.root')[0]),
        'lepton': np.full(len(slim_df), channel),
    }, syst.replace(';1', '')


def main(args):
    input_files = [
        ifile for ifile in glob('{}/*.root'.format(args.el_input_dir))
    ]
    input_files += [
        ifile for ifile in glob('{}/*.root'.format(args.mu_input_dir)) if args.mu_input_dir != None
    ]

    all_data = {}
    default_object = {
        'unscaled': pd.DataFrame(dtype='float64'),
        'selection': pd.DataFrame(),
        'names': np.array([]),
        'leptons': np.array([]),
        'isSignal': np.array([]),
        'weights': np.array([]),
        'index': np.array([]),
    }

    for ifile in input_files:
        open_file = uproot.open(ifile)
        for ikey in open_file.iterkeys():
            if not '_tree' in ikey:
                continue
            # TEMPORARY
            if 'tau_tree_jetVeto30_JetTotal' in ikey or 'JetTotal' in ikey or 'tree_Up' in ikey or 'tree_Down' in ikey:
                continue

            proc_file, syst = loadFile(ifile, open_file, ikey, args.category)
            all_data.setdefault(syst, default_object.copy())

            # add data to the full set
            all_data[syst]['unscaled'] = pd.concat([all_data[syst]['unscaled'], proc_file['slim_df']])
            # add selection variables to full set
            all_data[syst]['selection'] = pd.concat([all_data[syst]['selection'], proc_file['selection_df']])
            # insert the name of the current sample
            all_data[syst]['names'] = np.append(all_data[syst]['names'], proc_file['somenames'])
            # labels for signal/background
            all_data[syst]['isSignal'] = np.append(all_data[syst]['isSignal'], proc_file['isSignal'])
            # weights scaled from 0 - 1
            all_data[syst]['weights'] = np.append(all_data[syst]['weights'], proc_file['weights'])
            all_data[syst]['leptons'] = np.append(all_data[syst]['leptons'], proc_file['lepton'])  # lepton channel
            all_data[syst]['index'] = np.append(all_data[syst]['index'], proc_file['index'])

    # create the store
    store = pd.HDFStore('datasets/{}.h5'.format(args.output), complevel=9, complib='bzip2')

    # normalize the potential training variables
    scaler = StandardScaler()
    scaler.fit(all_data['tree']['unscaled'].values)  # only fit the nominal data
    scaler_info = pd.DataFrame.from_dict({
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'variance': scaler.var_,
        'nsamples': scaler.n_samples_seen_
    })
    scaler_info.set_index(all_data['tree']['unscaled'].columns.values, inplace=True)
    store['scaler'] = scaler_info  # save scaling info

    formatted_data = {}
    for syst in all_data.keys():
        # do the variable transform
        formatted_data[syst] = pd.DataFrame(
            scaler.transform(all_data[syst]['unscaled'].values),
            columns=all_data[syst]['unscaled'].columns.values, dtype='float64')

        # add selection variables
        for column in all_data[syst]['selection'].columns:
            formatted_data[syst][column] = all_data[syst]['selection'][column].values

        # add other useful data
        formatted_data[syst]['sample_names'] = pd.Series(all_data[syst]['names'])
        formatted_data[syst]['lepton'] = pd.Series(all_data[syst]['leptons'])
        formatted_data[syst]['isSignal'] = pd.Series(all_data[syst]['isSignal'])
        formatted_data[syst]['evtwt'] = pd.Series(all_data[syst]['weights'])
        formatted_data[syst]['idx'] = pd.Series(all_data[syst]['index'])
        store[syst] = formatted_data[syst]


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--el-input', '-e', action='store',
                        dest='el_input_dir', default=None, help='path to etau input files')
    parser.add_argument('--mu-input', '-m', action='store',
                        dest='mu_input_dir', default=None, help='path to mutau input files')
    parser.add_argument('--output', '-o', action='store', dest='output',
                        default='store.h5', help='name of output file')
    parser.add_argument('--category', '-c', action='store', dest='category',
                        default='vbf', help='name of category for selection')

    main(parser.parse_args())
