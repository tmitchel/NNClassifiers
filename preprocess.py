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
selection_vars = [
    'njets', 'OS',
    'is_signal', 'cat_vbf', 'nbjets',
    'mt', 't1_dmf', 't1_dmf_new', 't1_decayMode'
]

# Variables that could be used as NN input. These should be normalized
scaled_vars = [
    'evtwt', 'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1',
    'costheta2', 'costhetastar', 'mjj', 'higgs_pT', 'm_sv',
    't1_pt', 'MT_t2MET', 'MT_HiggsMET', 'jmet_dphi',
    'lt_dphi', 'el_pt', 'mu_pt', 'hj_dphi', 'MT_lepMET', 'met',
    'ME_sm_VBF', 'ME_bkg'
]

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
    columns = scaled_vars + selection_vars
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

    # combine lepton pT's
    if channel == 'mt':
        slim_df['lep_pt'] = slim_df['mu_pt'].copy()
    elif channel == 'et':
        slim_df['lep_pt'] = slim_df['el_pt'].copy()
    slim_df = slim_df.drop(['el_pt', 'mu_pt'], axis=1)

    # add Dbkg_VBF
    slim_df['Dbkg_VBF'] = slim_df['ME_sm_VBF'] / (45 * slim_df['ME_bkg'] + slim_df['ME_sm_VBF'])
    slim_df = slim_df.drop(['ME_sm_VBF', 'ME_bkg'], axis=1)

    # get variables needed for selection (so they aren't normalized)
    selection_df = slim_df[selection_vars]

    # get just the weights (they are scaled differently)
    weights = slim_df['evtwt']
    index = slim_df['idx']
    if 'vbf_inc' in ifile:
        weights = weights * slim_df['wt_a1']
    slim_df = slim_df.drop(selection_vars+todrop, axis=1)

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
        'syst': np.full(len(slim_df), syst.replace(';1', ''))
    }


def main(args):
    input_files = [
        ifile for ifile in glob('{}/*.root'.format(args.el_input_dir))
    ]
    input_files += [
        ifile for ifile in glob('{}/*.root'.format(args.mu_input_dir)) if args.mu_input_dir != None
    ]

    # define collections that will all be merged in the end
    unscaled_data, selection_df = pd.DataFrame(), pd.DataFrame()
    names, leptons, isSignal, weight_df, index, systs = np.array(
        []), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])

    for ifile in input_files:
        open_file = uproot.open(ifile)
        for ikey in open_file.iterkeys():
            if not '_tree' in ikey:
                continue
            # TEMPORARY
            if 'tau_tree_jetVeto30_JetTotal' in ikey:
                continue
            proc_file = loadFile(ifile, open_file, ikey, args.category)
            # add data to the full set
            unscaled_data = pd.concat([unscaled_data, proc_file['slim_df']])
            # add selection variables to full set
            selection_df = pd.concat([selection_df, proc_file['selection_df']])
            # insert the name of the current sample
            names = np.append(names, proc_file['somenames'])
            # labels for signal/background
            isSignal = np.append(isSignal, proc_file['isSignal'])
            # weights scaled from 0 - 1
            weight_df = np.append(weight_df, proc_file['weights'])
            leptons = np.append(leptons, proc_file['lepton'])  # lepton channel
            index = np.append(index, proc_file['index'])
            systs = np.append(systs, proc_file['syst'])

    # normalize the potential training variables
    scaled_data = pd.DataFrame(
        StandardScaler().fit_transform(unscaled_data.values),
        columns=unscaled_data.columns.values
    )

    # add selection variables
    for column in selection_df.columns:
        scaled_data[column] = selection_df[column].values

    # add other useful data
    scaled_data['sample_names'] = pd.Series(names)
    scaled_data['lepton'] = pd.Series(leptons)
    scaled_data['isSignal'] = pd.Series(isSignal)
    scaled_data['evtwt'] = pd.Series(weight_df)
    scaled_data['syst'] = pd.Series(systs)
    scaled_data['idx'] = pd.Series(index)

    # save the dataframe for later
    store = pd.HDFStore('datasets/{}.h5'.format(args.output))
    store['df'] = scaled_data


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
