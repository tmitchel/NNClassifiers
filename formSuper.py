import uproot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def build_background(files):
    bkg_df = pd.DataFrame([])
    for ifile in files:
        tree = uproot.open('{}'.format(ifile))['mutau_tree']
        df = tree.pandas.df(['NN_disc', 'm_sv'])
        df = df[(df['NN_disc'] > -1)]
        df['isSignal'] = np.zeros(len(df))
        bkg_df = pd.concat([bkg_df, df])
    return bkg_df

sig_tree = uproot.open('root_files/lda/ggh_madgraph_twojet.root')['mutau_tree']

sig = sig_tree.pandas.df(['NN_disc', 'm_sv'])
sig = sig[(sig['NN_disc'] > -1)]
sig['isSignal'] = np.ones(len(sig))
bkg = build_background(['root_files/lda/embedded.root'])

combined = pd.concat([sig, bkg])

lda = LinearDiscriminantAnalysis(n_components=1)
fitted = lda.fit(combined.values[:, 0:2], combined['isSignal'].values)
sig['super'] = fitted.transform(sig.values[:, 0:2])
bkg['super'] = fitted.transform(bkg.values[:, 0:2])

plt.scatter(sig['NN_disc'].values, sig['m_sv'].values, alpha=0.5, color='red')
plt.scatter(bkg['NN_disc'].values, bkg['m_sv'].values, alpha=0.5, color='blue')
plt.savefig('test2.png')

plt.figure(figsize=(30, 10))
ax = plt.subplot(1, 2, 1)
ax.hist(sig['super'].values, histtype='stepfilled', color='red',
         label='ggH', bins=50, density=True, alpha=0.5, range=(-2.5,3))
ax.hist(bkg['super'].values, histtype='stepfilled', color='blue',
        label='ZTT', bins=50, density=True, alpha=0.5, range=(-2.5, 3))
ax.legend(loc='best', shadow=False)

ax = plt.subplot(1, 2, 2)
plt.hist(sig['NN_disc'].values, histtype='stepfilled', color='red',
         label='ggH', bins=50, density=True, alpha=0.5)
plt.hist(bkg['NN_disc'].values, histtype='stepfilled', color='blue',
         label='ZTT', bins=50, density=True, alpha=0.5)
plt.legend(loc='best', shadow=False)
plt.savefig('test.png')
