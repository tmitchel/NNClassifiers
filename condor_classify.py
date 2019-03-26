import pandas as pd
from ROOT import TFile
from glob import glob
from array import array
from os import environ, path, mkdir
from multiprocessing import Process
environ['KERAS_BACKEND'] = 'tensorflow'
from keras.models import load_model

class Predictor:
  def __init__(self, data_name, model_name, keep):
    self.bad = False
    self.keep = keep
    self.data_copy = pd.DataFrame()
    # open the input data
    try:
      self.data = pd.HDFStore(data_name)['df']
    except:
      self.bad = True

    # open the trained model
    try:
      self.model = load_model('models/{}.hdf5'.format(model_name))
    except:
      self.bad = True

  def make_prediction(self, fname, channel):
    self.data_copy = self.data_copy.iloc[0:0]
    if not self.bad:
      self.data_copy = self.data[
              (self.data['sample_names'] == fname) & (self.data['lepton'] == channel)
          ].copy()

      to_classify = self.data_copy[self.keep]
      guesses = self.model.predict(to_classify.values, verbose=False)
      self.data_copy['guess'] = guesses
      self.data_copy.set_index('idx', inplace=True)

  def getGuess(self, index, syst='tree'):
    try:
      guess = self.current_tree.loc[index, 'guess']
    except:
      guess = -999
    return guess

  def setTree(self, syst='tree'):
    self.current_tree = self.data_copy[(self.data_copy['syst'] == syst)]

def fillFile(ifile, channel, args, vbf_pred, boost_pred):
  fname = ifile.split('/')[-1].split('.root')[0]
  print 'Starting process for file: {}'.format(fname)

  vbf_pred.make_prediction(fname, channel)
  boost_pred.make_prediction(fname, channel)

  root_file = TFile(ifile, 'READ')
  oname = ifile.split('/')[-1].split('.root')[0]
  fout = TFile('{}/{}.root'.format(args.output_dir, oname), 'recreate')  ## make new file for output
  fout.cd()
  nevents = root_file.Get('nevents').Clone()
  nevents.Write()
  
  ## now let's try and get this into the root file
  print 'Reading input file {}'.format(ifile)
  for ikey in root_file.GetListOfKeys():
    print ikey.GetName()
    if not '_tree' in ikey.GetName():
      continue
    itree = root_file.Get(ikey.GetName())
    syst_label = ikey.GetName().replace('mutau_', '')
    syst_label = syst_label.replace('etau_', '')
    vbf_pred.setTree(syst_label)
    # boost_pred.setTree(syst_label)

    fout.cd()
    ntree = itree.CloneTree(-1, 'fast')

    branch_var = array('f', [0.])
    branch_var_vbf = array('f', [0.])
    branch_var_boost = array('f', [0.])
    disc_branch = ntree.Branch('NN_disc', branch_var, 'NN_disc/F')
    disc_branch_vbf = ntree.Branch('NN_disc_vbf', branch_var_vbf, 'NN_disc_vbf/F')
    disc_branch_boost = ntree.Branch('NN_disc_boost', branch_var_boost, 'NN_disc_boost/F')
    nevts = ntree.GetEntries()
    
    evt_index = 0
    for _ in itree:
      if evt_index % 100000 == 0 and evt_index > 0:
        print 'Process: {} has completed: {} events out of {}'.format(fname, evt_index, nevts)
      branch_var[0] = vbf_pred.getGuess(evt_index)
      branch_var_vbf[0] = vbf_pred.getGuess(evt_index)
      branch_var_boost[0] = boost_pred.getGuess(evt_index)
      
      evt_index += 1
      fout.cd()
      disc_branch.Fill()
      disc_branch_vbf.Fill()
      disc_branch_boost.Fill()

    fout.cd()
    print 'writing tree {}'.format(ntree.GetName())
    ntree.Write()

  root_file.Close()
  fout.Close()
  print '{} Completed.'.format(fname)

def main(args):
    if args.treename == 'mutau_tree':
        channel = 'mt'
    elif args.treename == 'etau_tree':
        channel = 'et'
    else:
        raise Exception('Hey. Bad channel. No. Try again.')

    if not path.isdir(args.output_dir):
        mkdir(args.output_dir)

    file_names = []
    if args.input_dir != None and args.single_file == None:
      file_names = [ifile for ifile in glob('{}/*.root'.format(args.input_dir))]
    elif args.input_dir == None and args.single_file != None:
      file_names = [args.single_file]
    else:
      raise Exception('Can\'t use single file and full directory options together')

    keep_vbf = [
      'm_sv', 'mjj', 'higgs_pT', 'Q2V1', 'Q2V2', 'Phi', 
      'Phi1', 'costheta1', 'costheta2', 'costhetastar'
    ]
    vbf_pred = Predictor(args.input_vbf, args.model_vbf, keep_vbf)

    keep_boost = [
             'higgs_pT', 't1_pt', 'lt_dphi', 'lep_pt', 'hj_dphi', 'MT_lepMET', 'MT_HiggsMET', 'met'
    ]
    boost_pred = Predictor(args.input_boost, args.model_boost, keep_boost)

    processes = [Process(target=fillFile, args=(ifile, channel, args, vbf_pred, boost_pred)) for ifile in file_names]
    #for process in processes:
    #processes[0].start()
    [fillFile(ifile, channel, args, vbf_pred, boost_pred) for ifile in file_names]

    #print 'Finished processing.'


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--treename', '-t', action='store', dest='treename', default='etau_tree', help='name of input tree')
    parser.add_argument('--model-vbf', action='store', dest='model_vbf', default=None, help='name of model to use')
    parser.add_argument('--model-boost', action='store', dest='model_boost', default=None, help='name of model to use')
    parser.add_argument('--input-vbf', action='store', dest='input_vbf', default=None, help='name of input dataset')
    parser.add_argument('--input-boost', action='store', dest='input_boost', default=None, help='name of input dataset')
    parser.add_argument('--dir', '-d', action='store', dest='input_dir', default=None, help='name of ROOT input directory')
    parser.add_argument('--output-dir', '-o', action='store', dest='output_dir', default='output_files', help='name of directory for output')
    parser.add_argument('--file', '-f', action='store', dest='single_file', default='test.root', help='name of directory for output')

    main(parser.parse_args())
