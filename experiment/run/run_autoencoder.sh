#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --train-file ../data/datasets/ --val-file ../data/datasets/ "
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 43 --train-file ../data/datasets/ --val-file ../data/datasets/ "
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 44 --train-file ../data/datasets/ --val-file ../data/datasets/ "
echo $cmd >> log.txt
eval $cmd
