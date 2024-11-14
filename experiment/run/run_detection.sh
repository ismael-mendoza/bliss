#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

./bin/run_detection_train.py --seed 42 --ds-seed 42 --train-file ../data/datasets/train_ds_42.npz --val-file ../data/datasets/val_ds_42.npz
