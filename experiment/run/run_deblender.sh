#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"


./bin/run_deblender_train.py --seed 41 --ae-model-path ../models/autoencoder_42.pt \
--train-file ../data/datasets/train_ds_41_20240927143647.pt \
--val-file ../data/datasets/val_ds_41_20240927143647.pt
