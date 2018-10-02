import pandas
import numpy as np
import root_pandas as rp
from sklearn.metrics import roc_curve, auc

def getData(fname, isSignal):
    df = rp.read_root(fname)
    df = pandas.DataFrame(df[(df['cat_vbf'] > 0)]['NN_disc'])
    if isSignal:
        label = np.ones(len(df))
    else:
        label = np.zeros(len(df))
    df.insert(loc=df.shape[1], column='label', value=label)
    return df

sig = getData('output_files/etau_vbfcat/VBF125_NN.root', True)
bkg = getData('output_files/etau_vbfcat/ggH125_NN.root', False)
print 'sig', len(sig), 'bkg', len(bkg)
all_data = pandas.concat([sig, bkg])
dataset = all_data.values
data = dataset[:,0:1]
labels = dataset[:,-1]
print data, labels
fpr, tpr, thresholds = roc_curve(labels,  data[:, 0])

roc_auc = auc(fpr, tpr)

import  matplotlib.pyplot  as plt
plt.plot(tpr, fpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
plt.legend(loc="lower right")
plt.savefig('eh.pdf')
