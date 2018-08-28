#!/bin/bash

while getopts "vm:i:" OPTION
do
    case $OPTION in 
        m)
            model=$OPTARG
            ;;
        i)
            input=$OPTARG
            ;;
        v)
            verbose="--verbose"
            ;;
    esac
done

for i in $input/*.root
do
    #echo "python run_network $verbose -l $model -i $i"
    echo $i
    python run_network.py $verbose -l $model -i $i
done
