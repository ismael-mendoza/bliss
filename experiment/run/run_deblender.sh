#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"

./bin/run_deblender_train.py --seed 42 --ds-seed 42 --ae-model-path ../models/autoencoder_42_42.pt --train-file ../data/datasets/train_ds_42.npz --val-file ../data/datasets/val_ds_42.npz
