# import uproot
import ROOT as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from array import array
from root_pandas import read_root


def build_background(files):
    bkg_df = pd.DataFrame([])
    for ifile in files:
        df = read_root(ifile, columns=['NN_disc', 'm_sv', 'evtwt'])
        df['idx'] = np.array([i for i in xrange(0, len(df))])
        df = df[(df['NN_disc'] > -1)]
        df['isSignal'] = np.zeros(len(df))
        df['name'] = np.full(len(df), ifile.split('/')[-1].split('.root')[0])
        bkg_df = pd.concat([bkg_df, df])
    return bkg_df


def build_signal(files):
    sig_df = pd.DataFrame([])
    for ifile in files:
        df = read_root(ifile, columns=['NN_disc', 'm_sv', 'evtwt'])
        df['idx'] = np.array([i for i in xrange(0, len(df))])
        df = df[(df['NN_disc'] > -1)]
        df['isSignal'] = np.ones(len(df))
        df['name'] = np.full(len(df), ifile.split('/')[-1].split('.root')[0])
        sig_df = pd.concat([sig_df, df])
    return sig_df


def getDisc(df, index):
    try:
        superDisc = df.loc[index, 'super']
    except:
        superDisc = -999
    return superDisc


def insert(df, ifile, treename):
    root_file = rt.TFile('{}'.format(ifile), 'READ')
    itree = root_file.Get(treename)

    oname = ifile.split('/')[-1].split('.root')[0]
    fout = rt.TFile('{}/{}.root'.format(args.output_dir, oname),
                    'recreate')  # make new file for output
    fout.cd()
    nevents = root_file.Get('nevents').Clone()
    nevents.Write()
    ntree = itree.CloneTree(-1, 'fast')

    adiscs = array('f', [0.])
    disc_branch = ntree.Branch('super', adiscs, 'super/F')
    evt_index = 0
    for event in itree:
        if evt_index % 100000 == 0:
            print 'Processing: {}% completed'.format(
                (evt_index*100)/ntree.GetEntries())

        adiscs[0] = getDisc(df, evt_index)
        evt_index += 1
        fout.cd()
        disc_branch.Fill()
    root_file.Close()
    fout.cd()
    ntree.Write()
    fout.Close()


def doLDA(input_data, all_samples):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=1)
    fitted = lda.fit(input_data[['m_sv', 'NN_disc']].values, input_data['isSignal'].values)
    all_samples['super'] = fitted.transform(all_samples[['m_sv', 'NN_disc']].values)
    return all_samples


def doNN(input_data, all_samples):
    from os import environ
    environ['KERAS_BACKEND'] = 'tensorflow'
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.preprocessing import StandardScaler


    model = Sequential()
    model.add(Dense(1, input_shape=(2,), name='input', activation='sigmoid'))
    # model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])

    ## build callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        ModelCheckpoint('models/{}.hdf5'.format('test.hdf5'), monitor='val_loss',
                        verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto',
                        period=1
                        )
    ]

    scaled_input_data = pd.DataFrame(StandardScaler().fit_transform(input_data[['m_sv', 'NN_disc']].values), columns=input_data[['m_sv', 'NN_disc']].columns.values)
    input_data.loc[:, 'm_sv'] = scaled_input_data['m_sv'].values
    input_data.loc[:, 'NN_disc'] = scaled_input_data['NN_disc'].values
    scaled_all_samples = pd.DataFrame(StandardScaler().fit_transform(all_samples[['m_sv', 'NN_disc']].values), columns=all_samples[['m_sv', 'NN_disc']].columns.values)
    all_samples.loc[:, 'm_sv'] = scaled_all_samples['m_sv'].values
    all_samples.loc[:, 'NN_disc'] = scaled_all_samples['NN_disc'].values

    history = model.fit(input_data[['m_sv', 'NN_disc']].values, input_data['isSignal'].values, shuffle=True,
                        epochs=10000, batch_size=1024, verbose=True,
                        callbacks=callbacks, validation_split=0.25, sample_weight=input_data['evtwt'].values
                        )

    all_samples['super'] = model.predict(all_samples[['m_sv', 'NN_disc']].values)
    return all_samples

    

def main(args):
    prefix = '/afs/hep.wisc.edu/home/tmitchel/private/higgsToTauTau/ltau_analyzer/CMSSW_9_4_0/src/ltau_analyzers/Output/trees/mutau2016_official_v1p3_NNed/'

    sig_names = ['ggh_madgraph_twojet']
    bkg_names = ['embedded', 'TTT', 'TTJ', 'ZJ', 'ZL', 'VVJ', 'VVT', 'W']

    sig = build_signal(['{}/{}.root'.format(prefix, name)
                        for name in sig_names])
    bkg = build_background(['{}/{}.root'.format(prefix, name)
                            for name in bkg_names])

    input_data = pd.concat([sig, bkg])

    file_names = [ifile for ifile in glob('{}/*.root'.format(prefix))]
    all_samples = build_signal(file_names)
    all_samples.set_index('idx', inplace=True)

    if args.doLDA:
        all_samples = doLDA(input_data, all_samples)
    elif args.doNN:
        all_samples = doNN(input_data, all_samples)
    
    for iname in file_names:
        name = iname.split('/')[-1].split('.root')[0]
        insert(all_samples[(all_samples['name'] == name)],
               prefix+name+'.root', args.treename)



if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-t', '--treename', action='store',
                        dest='treename', help='name of tree')
    parser.add_argument('-d', '--output-dir', action='store',
                        dest='output_dir', help='name of output directory')
    parser.add_argument('-l', '--doLDA', action='store_true',
                        dest='doLDA', help='do Fisher Linear Discriminant Analysis')
    parser.add_argument('-n', '--doNN', action='store_true',
                        dest='doNN', help='do simple NN (single Perceptron)')
    args = parser.parse_args()
    main(args)
