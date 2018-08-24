#!/usr/bin/env bash

for file in *.root
do
  root2hdf5 -n 50000 $file
done
