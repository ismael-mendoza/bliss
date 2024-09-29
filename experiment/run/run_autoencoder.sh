#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="3"

./bin/run_autoencoder_train.py --seed 41 \
--train-file ../data/datasets/train_ae_ds_42_20240920164841.pt \
--val-file ../data/datasets/val_ae_ds_42_20240920164841.pt"
