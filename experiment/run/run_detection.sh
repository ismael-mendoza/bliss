#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="4"

./bin/run_detection_train.py --seed 22 --ds-seed 23 --train-file ../data/datasets/train_ds_detection_23.npz --val-file ../data/datasets/val_ds_detection_23.npz
