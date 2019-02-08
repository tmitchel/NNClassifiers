# import uproot
import ROOT as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from array import array
from root_pandas import read_root
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def build_background(files):
    bkg_df = pd.DataFrame([])
    for ifile in files:
        df = read_root(ifile, columns=['NN_disc', 'm_sv'])
        df['idx'] = np.array([i for i in xrange(0, len(df))])
        df = df[(df['NN_disc'] > -1)]
        df['isSignal'] = np.zeros(len(df))
        df['name'] = np.full(len(df), ifile.split('/')[-1])
        bkg_df = pd.concat([bkg_df, df])
    return bkg_df


def build_signal(files):
    sig_df = pd.DataFrame([])
    for ifile in files:
        df = read_root(ifile, columns=['NN_disc', 'm_sv'])
        df['idx'] = np.array([i for i in xrange(0, len(df))])
        df = df[(df['NN_disc'] > -1)]
        df['isSignal'] = np.ones(len(df))
        df['name'] = np.full(len(df), ifile.split('/')[-1])
        sig_df = pd.concat([sig_df, df])
    return sig_df


def getDisc(df, index):
    try:
        superDisc = df.loc[index, 'super']
    except:
        superDisc = -999
    return superDisc


def insert(df, ifile, treename):
    root_file = rt.TFile('{}.root'.format(ifile), 'READ')
    itree = root_file.Get(treename)

    oname = ifile.split('/')[-1].split('.root')[0]
    fout = rt.TFile('{}/{}.root'.format(args.output_dir, oname),
                    'recreate')  # make new file for output
    fout.cd()
    nevents = root_file.Get('nevents').Clone()
    nevents.Write()
    ntree = itree.CloneTree(-1, 'fast')

    adiscs = array('f', [0.])
    disc_branch = ntree.Branch('NN_disc', adiscs, 'NN_disc/F')
    evt_index = 0
    for event in itree:
        if evt_index % 10000 == 0:
            print 'Processing: {}% completed'.format(
                (evt_index*100)/ntree.GetEntries())

        adiscs[0] = getDisc(sample, evt_index)
        evt_index += 1
        fout.cd()
        disc_branch.Fill()
    root_file.Close()
    fout.cd()
    ntree.Write()
    fout.Close()


def plot(sig, bkg):
    plt.scatter(bkg['NN_disc'].values, bkg['m_sv'].values,
                alpha=0.5, color='blue')
    plt.scatter(sig['NN_disc'].values, sig['m_sv'].values,
                alpha=0.5, color='red')
    plt.savefig('test2.pdf')

    plt.figure(figsize=(30, 10))
    ax = plt.subplot(1, 2, 1)
    ax.hist(sig['super'].values, histtype='stepfilled', color='red',
            label='ggH', bins=50, normed=True, alpha=0.5, range=(-2.5, 3))
    ax.hist(bkg['super'].values, histtype='stepfilled', color='blue',
            label='ZTT', bins=50, normed=True, alpha=0.5, range=(-2.5, 3))
    ax.legend(loc='best', shadow=False)

    ax = plt.subplot(1, 2, 2)
    plt.hist(sig['NN_disc'].values, histtype='stepfilled', color='red',
             label='ggH', bins=50, normed=True, alpha=0.5)
    plt.hist(bkg['NN_disc'].values, histtype='stepfilled', color='blue',
             label='ZTT', bins=50, normed=True, alpha=0.5)
    plt.legend(loc='best', shadow=False)
    plt.savefig('test.pdf')


def main(args):
    prefix = '~/private/higgsToTauTau/ltau_analyzer/CMSSW_9_4_0/src/ltau_analyzers/Output/trees/mutau2016_official_v1p3_NNed/'

    sig_names = ['vbf_inc']
    bkg_names = ['embedded', 'TTT', 'TTJ', 'ZJ', 'ZL', 'VVJ', 'VVT', 'W']

    sig = build_signal(['{}/{}.root'.format(prefix, name)
                        for name in sig_names])
    bkg = build_background(['{}/{}.root'.format(prefix, name)
                            for name in bkg_names])

    combined = pd.concat([sig, bkg])
    combined.set_index('idx', inplace=True)

    lda = LinearDiscriminantAnalysis(n_components=1)
    fitted = lda.fit(combined.values[:, 0:2], combined['isSignal'].values)
    sig['super'] = fitted.transform(sig.values[:, 0:2])
    bkg['super'] = fitted.transform(bkg.values[:, 0:2])

    for iname in sig_names + bkg_names:
        insert(combined[(combined['name'] == iname)],
               prefix+iname, args.treename)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-s', '--signal', action='store',
                        dest='signal', help='signal name')
    parser.add_argument('-t', '--treename', action='store',
                        dest='treename', help='name of tree')
    parser.add_argument('-d', '--output-dir', action='store',
                        dest='output_dir', help='name of output directory')
    args = parser.parse_args()
    main(args)
