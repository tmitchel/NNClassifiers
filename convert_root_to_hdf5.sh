#!/usr/bin/env bash

pushd ${1}
for file in *.root
do
  root2hdf5 -n 50000 $file
done
popd