#!/usr/bin/env bash

pushd ../input_files
for file in *.root
do
  root2hdf5 -n 50000 $file
done
popd
