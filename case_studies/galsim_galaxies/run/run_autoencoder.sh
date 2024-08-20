#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="7"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 44 --lr 1e-4 -t "43_1""
echo $cmd >> log.txt
eval $cmd
