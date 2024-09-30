#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

./bin/run_binary_train.py --seed 42 --train-file ../data/datasets/train_ds_42.hdf5 --val-file ../data/datasets/val_ds_42.hdf5
