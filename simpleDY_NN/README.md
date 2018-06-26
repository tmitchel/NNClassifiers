# Neural Network for separating VBF and Z->TT + N jets

## Instructions to setup the network

In order to use the framework, you must setup a virtual environment with all necessary packages installed

    source setup.sh

Once you have setup the virtual environment, you must activate the environment every time you login

    source pyenv/bin/activate

Then, you must convert all root files to to HDF5 files which are easily loaded by the network. Files will retain the same name, but different extension.

    bash convert_root_to_hdf5.sh

## Instructions about running the network

The network has many command line options. To print out all options, use

    python simple_net.py -h

Recommended first run

    python simple_net.py -v Q2V1 Q2V2 Phi Phi1 costhetastar costheta2 costheta1 -n 7 --verbose --input VBFHtoTauTau125_svFit_MELA --retrain

This command will train a neural network with one hidden layer containing 7 nodes. The network will be retrained using [Q2V1, Q2V2, Phi, Phi1, costhetastar, costheta2, costheta1] as input variables even if there is a model_checkpoint.hdf5 file. Then, all events from VBFHtoTauTau125_svFit_MELA.root will be processed by the NN and the output discriminant will be stored in a new root file.

The output root file with the NN discriminant included is

    VBFHtoTauTau125_svFit_MELA_NN.root

The HDF5 file to store the weights from the trained network is

    model_checkpoint.hdf5

## To-do

1. Add more in-code documentation
2. Refactor code into an easier to understand form
3. Fix code for putting NN discriminant in root files (it's pretty hacky right now)
