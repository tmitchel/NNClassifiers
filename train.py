from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense
from keras.models import Sequential
from keras import optimizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualize import *

def main(args):
    data = pd.HDFStore(args.input)['df']
    training_variables = [
        'Q2V1', 'Q2V2', 'Phi', 'Phi1', 'costheta1',
        'costheta2', 'costhetastar', 'isSignal', 'evtwt'
    ]

    model = Sequential()
    model.add(Dense(7, input_shape=(7,), name='input', activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid', kernel_initializer='normal'))
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy',
                metrics=['accuracy'])

    ## build callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        ModelCheckpoint('models/{}.hdf5'.format(args.model), monitor='val_loss',
                        verbose=0, save_best_only=True,
                        save_weights_only=False, mode='auto',
                        period=1
                        )
    ]

    ## get the data for the two-classes to discriminate
    training_processes = data[
        (data['sample_names'] == args.signal) | (data['sample_names'] == args.background)
    ]

    etau   = training_processes[(training_processes['lepton'] == 'et')]
    mutau  = training_processes[(training_processes['lepton'] == 'mt')]
    tautau = training_processes[(training_processes['lepton'] == 'tt')]

    ## do event selection
    selected_et, selected_mt, selected_tt = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    ## electron-tau channel
    if len(etau) > 0:
        selected_et = etau[
            (etau['cat_vbf'] > 0) &
            (etau['nbjets'] == 0) &
            (etau['el_charge'] + etau['t1_charge'] == 0)
        ]

    ## muon-tau channel
    if len(mutau) > 0:
        selected_mt = mutau[
            (mutau['cat_vbf'] > 0) &
            (mutau['nbjets'] == 0) &
            (mutau['mu_charge'] + mutau['t1_charge'] == 0)
        ]

    # ## tau-tau channel
    # if len(tautau) > 0:
    #     selected_tt = tautau[
    #         (training_processes['cat_vbf'] > 0) &
    #         (training_processes['nbjets'] == 0) &
    #         (training_processes['t1_charge'] +
    #         training_processes['t2_charge'] == 0)
    #     ]

    ## combine channels into total dataset
    selected_events = pd.concat([selected_et, selected_mt, selected_tt])

    training_dataframe = selected_events[training_variables]

    training_data, testing_data, training_meta, testing_meta = train_test_split(
        training_dataframe.values[:, :7], training_dataframe.values[:, 7:], test_size=0.1, random_state=7
    )

    training_labels = training_meta[:, 0]
    training_weights = training_meta[:, 1]

    ## train that there model, my dude
    history = model.fit(training_data, training_labels, shuffle=True,
                        epochs=10000, batch_size=1024, verbose=True,
                        callbacks=callbacks, validation_split=0.25, sample_weight=training_weights
                        )

    if not args.dont_plot:
        ROC_curve(training_data, training_labels, training_weights, model, 'ROC_training_{}'.format(args.model), 'red')
        ROC_curve(testing_data, testing_meta[:, 0], testing_meta[:, 1], model, 'ROC_testing_{}'.format(args.model), 'cyan')

        trainingPlots(history, 'trainingPlot_{}'.format(args.model))

        test_sig, test_bkg = [], []
        for i in range(len(testing_meta)):
            if testing_meta[i, 0] == 1:
                test_sig.append(testing_data[i, :])
            elif testing_meta[i, 0] == 0:
                test_bkg.append(testing_data[i, :])

        train_sig, train_bkg = [], []
        for i in range(len(training_meta)):
            if training_meta[i, 0] == 1:
                train_sig.append(training_data[i, :])
            elif training_meta[i, 0] == 0:
                train_bkg.append(training_data[i, :])

        discPlot('NN_disc_{}'.format(args.model), model, np.array(train_sig), np.array(train_bkg), np.array(test_sig), np.array(test_bkg))

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--model', '-m', action='store', dest='model', default='testModel', help='name of the model to train')
    parser.add_argument('--input', '-i', action='store', dest='input', default='test', help='full name of input file')
    parser.add_argument('--signal', '-s', action='store', dest='signal', default='VBF125.root', help='name of signal file')
    parser.add_argument('--background', '-b', action='store', dest='background', default='ZTT.root', help='name of background file')
    parser.add_argument('--dont-plot', action='store_true', dest='dont_plot', help='don\'t make training plots')

    main(parser.parse_args())
