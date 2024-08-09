#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -o -s 43 -t "1_43""
echo $cmd >> log.txt
eval $cmd
