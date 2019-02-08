#!/usr/bin bash

cd ..
if [ ! -d "$.pyenv" ]; then
  virtualenv .pyenv # setup virtual enviroment
fi
source .pyenv/bin/activate # activate.csh for tcsh
eval `scramv1 runtime -sh` # cmsenv

pip install --no-cache-dir --upgrade numpy==1.15.4
pip install --no-cache-dir matplotlib
pip install --no-cache-dir scikit-learn
pip install --no-cache-dir tables
pip install --no-cache-dir pandas
pip install --no-cache-dir tensorflow
pip install --no-cache-dir keras
pip install --no-cache-dir root_pandas
