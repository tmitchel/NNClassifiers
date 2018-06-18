#!/usr/bin bash

if [ ! -d "$pyenv" ]; then
  virtualenv pyenv # setup virtual enviroment
fi
source pyenv/bin/activate # activate.csh for tcsh
eval `scramv1 runtime -sh` # cmsenv

pip install keras
pip install tables
pip install pandas
