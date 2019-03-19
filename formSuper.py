# import uproot
import ROOT as rt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from array import array
from root_pandas import read_root


def build_dataframe(files, isSignal):
    apply_df = pd.DataFrame([])
    columns = [
        'NN_disc_vbf', 'm_sv', 'evtwt',
        'njets', 'mjj', 'is_signal', 'mt', 'OS'
    ]

    if isSignal:
        signal_array = np.ones
    else:
        signal_array = np.zeros

    for ifile in files:
        df = read_root(ifile, columns=columns)
        df['idx'] = np.array([i for i in xrange(0, len(df))])
        df = df[(df['NN_disc_vbf'] > -1)]
        df['isSignal'] = signal_array(len(df))
        df['name'] = np.full(len(df), ifile.split('/')[-1].split('.root')[0])
        apply_df = pd.concat([apply_df, df])
    
    # vbf category selection
    train_df = apply_df[
        (apply_df['is_signal'] > 0) & (apply_df['OS'] > 0) &
        (apply_df['njets'] > 1) & (apply_df['mjj'] > 300) & (apply_df['mt'] < 50)
    ]

    return train_df, apply_df


def getDisc(df, index):
    try:
        superDisc = df.loc[index, 'super']
    except:
        superDisc = -999
    return superDisc


def insert(df, ifile, treename):
    df.set_index('idx', inplace=True)
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


def doLDA(train_data, apply_data):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


    lda = LinearDiscriminantAnalysis(n_components=1)
    fitted = lda.fit(train_data[['m_sv', 'NN_disc_vbf']].values, train_data['isSignal'].values)
    apply_data['super'] = fitted.transform(apply_data[['m_sv', 'NN_disc_vbf']].values)
    return apply_data


def doNN(train_data, apply_data):
    import visualize
    from os import environ
    environ['KERAS_BACKEND'] = 'tensorflow'
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.callbacks import ModelCheckpoint, EarlyStopping
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split


    model = Sequential()
    model.add(Dense(4, input_shape=(2,), name='input', activation='relu'))
    model.add(Dense(1, activation='relu'))
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

    scaled_train_data = pd.DataFrame(StandardScaler().fit_transform(train_data[['m_sv', 'NN_disc_vbf']].values), columns=train_data[['m_sv', 'NN_disc_vbf']].columns.values)
    scaled_apply_data = pd.DataFrame(StandardScaler().fit_transform(apply_data[['m_sv', 'NN_disc_vbf']].values), columns=apply_data[['m_sv', 'NN_disc_vbf']].columns.values)
    scaled_train_data['evtwt'] = MinMaxScaler(
        feature_range=(1., 2.)
    ).fit_transform(
        train_data['evtwt'].values.reshape(-1, 1)
    )
    scaled_train_data['isSignal'] = train_data['isSignal'].values
    scaled_apply_data['name'] = apply_data['name'].values
    scaled_apply_data['idx'] = apply_data['idx'].values

    training_data, _, training_labels, _, training_weights, _  = train_test_split(
        scaled_train_data[['m_sv', 'NN_disc_vbf']].values, scaled_train_data['isSignal'].values, scaled_train_data['evtwt'].values,
        test_size=0, random_state=7
    )

    history = model.fit(training_data, training_labels, shuffle=True,
                        epochs=10000, batch_size=512, verbose=True,
                        callbacks=callbacks, validation_split=0.25, sample_weight=training_weights
                        )
    visualize.trainingPlots(history, 'trainingPlot_Perceptron')

    scaled_apply_data['super'] = model.predict(scaled_apply_data[['m_sv', 'NN_disc_vbf']].values)
    print scaled_apply_data[(scaled_apply_data['name'] == 'vbf_inc')]['super']
    return scaled_apply_data


def main(args):
    prefix = './output_files/mutau2016_official_v1p5-no-msv/'

    sig_names = ['ggh_madgraph_twojet']
    bkg_names = ['embedded', 'TTT', 'TTJ', 'ZJ', 'ZL', 'VVJ', 'VVT', 'W']
    others = [ifile.split('/')[-1].split('.root')[0] for ifile in glob('{}/*.root'.format(prefix))]
    for sample in sig_names + bkg_names:
        others.remove(sample)

    train_sig, apply_sig = build_dataframe(['{}/{}.root'.format(prefix, name)
                        for name in sig_names], True)
    train_bkg, apply_bkg = build_dataframe(['{}/{}.root'.format(prefix, name)
                            for name in bkg_names], False)
    train_other, apply_other = build_dataframe(['{}/{}.root'.format(prefix, name)
                            for name in others], False)

    train_data = pd.concat([train_sig, train_bkg])
    apply_data = pd.concat([apply_sig, apply_bkg, apply_other])

    if args.doLDA:
        filled_data = doLDA(train_data, apply_data)
    elif args.doNN:
        filled_data = doNN(train_data, apply_data)
    
    file_names = [ifile for ifile in glob('{}/*.root'.format(prefix))]
    for iname in file_names:
        name = iname.split('/')[-1].split('.root')[0]
        print name
        insert(filled_data[(filled_data['name'] == name)],
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
