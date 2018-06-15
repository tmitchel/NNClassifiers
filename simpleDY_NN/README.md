# Neural Network for separating VBF and Z->TT + 2 jets

### Instructions to setup/run the network

1.) Setup the python environment

    source setup.sh
This will create a python virtualenv and install keras. You only need to run this command the first time you setup a new directory. To use the environment

    source pyenv/bin/activate

2.) Get the data into a form that is easy to use

    python format_data.py

3.) Run the network

    python simple_net.py
