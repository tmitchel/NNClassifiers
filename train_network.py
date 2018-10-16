##############################
## Just the plotting things ##
##############################

def ROC_curve(data_test, label_test, weights, model):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    label_predict = model.model.predict(data_test)
    fpr, tpr, thresholds = roc_curve(
        label_test, label_predict[:, 0], sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2,
             color='k', label='random chance')
    plt.plot(tpr, fpr, lw=2, color='cyan', label='NN auc = %.3f' % (roc_auc))
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('true positive rate')
    plt.ylabel('false positive rate')
    plt.title('receiver operating curve')
    plt.legend(loc="upper left")
    plt.grid()
    plt.savefig('plots/ROC_'+model.name+'.pdf')


def trainingPlots(history, model):
    import matplotlib.pyplot as plt
    # plot loss vs epoch
    ax = plt.subplot(2, 1, 1)
    ax.plot(history.history['loss'], label='loss')
    ax.plot(history.history['val_loss'], label='val_loss')
    ax.legend(loc="upper right")
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')

    # plot accuracy vs epoch
    ax = plt.subplot(2, 1, 2)
    ax.plot(history.history['acc'], label='acc')
    ax.plot(history.history['val_acc'], label='val_acc')
    ax.legend(loc="upper left")
    ax.set_xlabel('epoch')
    ax.set_ylabel('acc')
    plt.savefig('plots/trainingPlot_'+model.name+'.pdf')


def discPlot(model, sig, bkg):
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import StandardScaler
    sig = sig.values[:, 0:model.ninp]
    bkg = bkg.values[:, 0:model.ninp]

    sig = StandardScaler().fit_transform(sig)
    bkg = StandardScaler().fit_transform(bkg)

    sig_pred = model.model.predict(sig)
    bkg_pred = model.model.predict(bkg)

    plt.figure(figsize=(12, 8))
    plt.title('NN Discriminant')
    plt.xlabel('NN Disc.')
    plt.ylabel('Events/Bin')
    plt.hist(bkg_pred, histtype='step', color='red', label='ZTT', bins=100)
    plt.hist(sig_pred, histtype='step', color='blue', label='VBF', bins=100)
    plt.legend()
    plt.savefig('plots/disc_'+model.name+'.pdf')

##############################
## End the plotting things  ##
##############################

import numpy as np
import pandas as pd
from glob import glob
from os import environ
from root_pandas import read_root
# on Wisc machine, must be before Keras import
environ['KERAS_BACKEND'] = 'tensorflow'
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping

def create_json(model_name, nhid, vars):
    import json

    if model_name != None:
        fname = model_name + '.json'
    else:
        fname = 'model_store.json'

    with open('model_params/'+fname, 'w') as fout:
        json.dump(
            {
                'model_name': model_name,
                'variables': vars,
                'nhidden': nhid[0],
                'n_user_inputs': len(vars)
            }, fout
        )

class Classifier:
    def __init__(self, name, ninp, nhid):

        # initialize data holders
        self.name = name
        self.ninp = ninp
        self.sig = pd.DataFrame()
        self.bkg = pd.DataFrame()
        self.data = NotImplemented
        self.label = NotImplemented
        self.weight = NotImplemented
        self.callbacks = NotImplemented

        # first, build the model
        self.model = Sequential()

        # add the input layer
        self.model.add(
            Dense(nhid[0], input_shape=(nhid[0],), name='input')
        )

        # add hidden layers
        for i in range(len(nhid)-1):
            self.model.add(
                Dense(nhid[i], activation='sigmoid', kernel_initializer='normal')
            )

        # last, add output layer
        self.model.add(
            Dense(1, activation='sigmoid', kernel_initializer='normal')
        )

        # print information while training
        self.model.summary()

        # then, compile the model
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy'])

        # now add some callbacks
        self.callbacks = [
            EarlyStopping(monitor='val_loss', patience=50),
            ModelCheckpoint('models/'+name+'.hdf5', monitor='val_loss',
                            verbose=0, save_best_only=True,
                            save_weights_only=False, mode='auto',
                            period=1
                            )
        ]

    def loadData(self, vars, sig_name, bkg_name):
        # quick function for loading
        def load(name):
            other_vars = [
                'evtwt', 'cat_inclusive', 'cat_0jet', 'cat_boosted', 'cat_vbf',
                'Dbkg_VBF', 'Dbkg_ggH', 'njets', 'higgs_pT', 't1_charge', 'el_charge'
            ]
            slicer = vars + other_vars  # add variables for event selection
            df = read_root(name, columns=slicer)

            # apply selection and make sure variables are reasonable
            qual_cut = (df['Q2V1'] > 0) & \
                       (df['cat_vbf'] > 0) & \
                       (df['el_charge'] + df['t1_charge'] == 0)

            df = df[qual_cut]

            # make sure the weight is in the correct column
            weight = df['evtwt'].values
            from sklearn.preprocessing import MinMaxScaler
            points = weight.shape[0]
            weight = weight.reshape(-1, 1)
            weight = MinMaxScaler(feature_range=(0, 1)).fit_transform(weight)
            weight = weight.reshape(points, -1)
            df.insert(loc=df.shape[1], column='weight', value=weight)

            # remove un-needed columns
            df = df.drop(other_vars, axis=1)

            return df

        self.sig = load(sig_name)
        self.bkg = load(bkg_name)

        self.sig['isSignal'] = np.ones(len(self.sig))
        self.bkg['isSignal'] = np.zeros(len(self.bkg))

        print 'Training Statistics ----'
        print 'No. Signal', self.sig.shape[0]
        print 'No. Backg.', self.bkg.shape[0]

    def buildTrainingSet(self):
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        fat_panda = pd.concat([self.sig, self.bkg])
        self.data = fat_panda.values

        # split data into labels and also split into train/test
        data_train, data_test, meta_train, meta_test = train_test_split(
            self.data[:, :self.ninp], self.data[:, self.ninp:], test_size=0.05, random_state=7)

        # normalize all input variables to improve performance
        data_train = StandardScaler().fit_transform(data_train)

        self.data = data_train
        self.weight = meta_train[:, 0]
        self.label = meta_train[:, 1]
        return data_test, meta_test

    def trainModel(self):
        return self.model.fit(self.data, self.label, shuffle=True,
                      epochs=10000, batch_size=1024, verbose=True,
                      callbacks=self.callbacks, validation_split=0.25, sample_weight=self.weight
                      )


def main(args):
    cl = Classifier(args.model_name, len(args.vars), args.nhid)
    cl.loadData(args.vars, args.signal, args.background)
    data_test, meta_test = cl.buildTrainingSet()

    history = cl.trainModel()

    ROC_curve(np.concatenate((data_test, cl.data), axis=0),
              np.concatenate((meta_test[:, 1], cl.label), axis=0),
              np.concatenate((meta_test[:, 0], cl.weight), axis=0),
              cl
              )

    if args.verbose:
        trainingPlots(history, cl)
        discPlot(cl, cl.sig, cl.bkg)

    if args.model_name != None:
        model_name = args.model_name
    else:
        model_name = 'NN_model'

    create_json(model_name, args.nhid, args.vars)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Train two-layer neural network to separate Drell-Yan + jets from VBF Higgs to tau-tau"
    )
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose', default=False,
                        help='run in verbose mode'
                        )
    parser.add_argument('--nhid', '-n', nargs='+', action='store',
                        dest='nhid', default=[7], type=int,
                        help='[# hidden, ...# nodes in layer]'
                        )
    parser.add_argument('--vars', '-v', nargs='+', action='store',
                        dest='vars', default=['Q2V1', 'Q2V2'],
                        help='variables to input to network'
                        )
    parser.add_argument('--model_name', '-m', action='store',
                        dest='model_name', default=None,
                        help='name of a trained model'
                        )
    parser.add_argument('--signal', action='store',
                        dest='signal', default='input_files/VBF125.root',
                        help='name of the signal file'
                        )
    parser.add_argument('--background', action='store',
                        dest='background', default='input_files/embed.root',
                        help='name of background file'
                        )
    main(parser.parse_args())
