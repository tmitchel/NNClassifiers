import h5py
import pandas
import numpy as np
from sklearn.metrics import roc_curve, auc

# np.set_printoptions(threshold=np.nan)

def getData(fname, isSignal):
    ifile = h5py.File('input_files/'+fname, 'r')
    ibranch = ifile['tt_tree'][('Dbkg_VBF', "numGenJets", "njets", "pt_sv", "jeta_1", "jeta_2", "againstElectronVLooseMVA6_1", "againstElectronVLooseMVA6_2", \
    "againstMuonLoose3_1", "againstMuonLoose3_2", "byTightIsolationMVArun2v1DBoldDMwLT_2", "byTightIsolationMVArun2v1DBoldDMwLT_1", "extraelec_veto", "extramuon_veto",\
    "byLooseIsolationMVArun2v1DBoldDMwLT_2", "byLooseIsolationMVArun2v1DBoldDMwLT_1", "mjj", 'Q2V1')]
    df = pandas.DataFrame(ibranch)
    ifile.close()

    df = df[(df['Dbkg_VBF'] > -100) ]

    sig_cuts = (df['Q2V1'] > -100) & (df['pt_sv'] > 100) & (df['mjj'] > 300) & (df['againstElectronVLooseMVA6_1'] > 0.5) & (df['againstElectronVLooseMVA6_2'] > 0.5) \
    & (df['againstMuonLoose3_1'] > 0.5) & (df['againstMuonLoose3_2'] > 0.5) & (df['byTightIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) & (df['byTightIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) \
    & (df['extraelec_veto'] < 0.5) & (df['extramuon_veto'] < 0.5) & ( (df['byLooseIsolationMVArun2v1DBoldDMwLT_1'] > 0.5) | (df['byLooseIsolationMVArun2v1DBoldDMwLT_2'] > 0.5) )
    if not isSignal:
      bkg_cuts = sig_cuts
      df = df[bkg_cuts]
      df['isSignal'] = np.zeros(len(df))
    else:
        df = df[sig_cuts]
        df['isSignal'] = np.ones(len(df))
    return df

sig = getData('VBFHtoTauTau125_svFit_MELA.h5', True)
bkg = getData('DY.h5', False)
print 'sig', len(sig), 'bkg', len(bkg)
all_data = pandas.concat([sig, bkg])
dataset = all_data.values
data = dataset[:,0:1]
labels = dataset[:,-1]
fpr, tpr, thresholds = roc_curve(labels, data)

for i in tpr:
  if i < 0.851 and i > 0.849:
    print i

ind = np.where(thresholds==0.8498852849006653)
print fpr[ind[0]], tpr[ind[0]]

print

ind = np.where(tpr==0.849640933572711)
print fpr[ind[0]], tpr[ind[0]]

roc_auc = auc(fpr, tpr)

import  matplotlib.pyplot  as plt
plt.plot(tpr, fpr, lw=2, color='cyan', label='auc = %.3f' % (roc_auc))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', label='random chance')
plt.legend(loc="lower right")
plt.show()
