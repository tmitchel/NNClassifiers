# Instructions to setup the network

In order to use the framework, you must setup a virtual environment with all necessary packages installed and convert all input root files to h5 file format. The tools to setup the framework are located in the build directory.

    cd build

1. Setup the virtual enviroment and install all necessary python packages

        source setup.sh

    Once you have setup the virtual environment, you must activate the environment every time you login. Navigate to the top of the framework

        tmitchel@pop-os:~/Documents/HTT_NN$ pwd
        /home/tmitchel/Documents/HTT_NN

    and activate the environmnet

        source pyenv/bin/activate

2. Then, you must convert all root files to to HDF5 files which are easily loaded by the network. Files will retain the same name, but different extension.

        bash convert_root_to_hdf5.sh

    This will convert all root files in the input_files directory into h5 files to be used by the network

# Instructions for running the network

The project will take input root files and produce new root files containing all old branches in addition to the NN discriminant. You must either produce your own trained network or copy an h5 file containing the trained network into the models directory. Two models are included in the models directory to test the code.

## Training your own network

The script train_network.py is used to create an h5 file with a trained model. The output will be stored in the models directory

An example usage is shown below

    python train_network.py -v Q2V1 Q2V2 Phi Phi1 costhetastar costheta2 costheta1 -n 7 --verbose

This will train a neural network with 1 hidden layer containing 7 nodes to separate H->TT from DY+2-Jets. The network will take [Q2V1, Q2V2, Phi, Phi1, costhetastar, costheta2, costheta1] as inputs and run printing out progress as it trains.

More command-line options exist for convenince. They can be seen with 

    python train_network.py -h

When run in verbose mode, the script will also produce a ROC curve saved as a pdf in the plots directory

## Running the network on an input file

The script run_network.py is used to process an input file using a pre-trained network. The output will be a new root file stored in the output_files directory.

An example usage is shown below

    python run_network.py -i DY -l my_network.json

This will load the model parameters (including the name) from my_network.json and run the trained network on an input file named DY.h5. The output file will be output_files/DY_NN.root

More command-line options exist for convenince. They can be seen with 

    python run_network.py -h

# To-do

1. Fix code for putting NN discriminant in root files (it's pretty hacky right now)

# (DEPRECATED) Instructions about running the network

The network has many command line options. To print out all options, use

    python simple_net.py -h

Recommended first run

    python simple_net.py -v Q2V1 Q2V2 Phi Phi1 costhetastar costheta2 costheta1 -n 7 --verbose --input VBFHtoTauTau125_svFit_MELA --retrain

This command will train a neural network with one hidden layer containing 7 nodes. The network will be retrained using [Q2V1, Q2V2, Phi, Phi1, costhetastar, costheta2, costheta1] as input variables even if there is a model_checkpoint.hdf5 file. Then, all events from VBFHtoTauTau125_svFit_MELA.root will be processed by the NN and the output discriminant will be stored in a new root file.

The output root file with the NN discriminant included is

    VBFHtoTauTau125_svFit_MELA_NN.root

The HDF5 file to store the weights from the trained network is

    model_checkpoint.hdf5

