#!/usr/bin bash

cd ..
if [ ! -d "$pyenv" ]; then
  virtualenv pyenv # setup virtual enviroment
fi
source pyenv/bin/activate # activate.csh for tcsh
eval `scramv1 runtime -sh` # cmsenv

pip install matplotlib
pip install -U scikit-learn
pip install tables
pip install pandas
pip install tensorflow
pip install keras
pip install root_pandas
