#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="1"

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-4 -t "44_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-5 -t "44_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 5e-4 -t "44_1""
echo $cmd >> log.txt
eval $cmd

echo >> log.txt
cmd="./bin/run_autoencoder_train.py -s 42 --lr 1e-3 -t "44_1""
echo $cmd >> log.txt
eval $cmd
