import h5py
import pandas
import numpy as np
from sklearn.metrics import roc_curve, auc

def getData(fname, isSignal):
    ifile = h5py.File('input_files/'+fname, 'r')
    ibranch = ifile['tt_tree'][('Dbkg_VBF')]
    df = pandas.DataFrame(ibranch)
    ifile.close()

    df = df[(df[0] > -100)]
    if isSignal:
        df['isSignal'] = np.ones(len(df))
    else:
        df['isSignal'] = np.zeros(len(df))

    return df

sig = getData('VBFHtoTauTau125_svFit_MELA.h5', True)
bkg = getData('DYJets2_svFit_MELA.h5', False)
all_data = pandas.concat([sig, bkg])
dataset = all_data.values
data = dataset[:,0:1]
labels = dataset[:,1]
fpr, tpr, thresholds = roc_curve(labels, data)
roc_auc = auc(fpr, tpr)

import  matplotlib.pyplot  as plt
plt.plot(tpr, fpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
plt.legend(loc="lower right")
plt.show()
