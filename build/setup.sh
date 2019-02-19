#!/usr/bin bash

cd ..
if [ ! -d "$.pyenv" ]; then
  virtualenv .pyenv # setup virtual enviroment
fi
source .pyenv/bin/activate # activate.csh for tcsh
eval `scramv1 runtime -sh` # cmsenv

pip install --no-cache-dir --user --upgrade numpy==1.15.4
pip install --no-cache-dir --user matplotlib
pip install --no-cache-dir --user scikit-learn
pip install --no-cache-dir --user tables
pip install --no-cache-dir --user pandas
pip install --no-cache-dir --user tensorflow
pip install --no-cache-dir --user keras
pip install --no-cache-dir --user root_pandas
pip install --no-cache-dir --user uproot
