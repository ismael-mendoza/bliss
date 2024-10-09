#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0"
export SEED="44"
export AE_VERSION="25" # find next one in "out/autoencoder" folder

../get_single_galaxies_datasets.py --seed $SEED

../get_blends_datasets.py --seed $SEED

./run_autoencoder_train.py --seed $SEED --ds-seed $SEED --train-file ../data/datasets/train_ae_ds_${SEED}.npz --val-file ../data/datasets/val_ae_ds_${SEED}.npz

../get_model_from_checkpoint.py -m "autoencoder" --seed $SEED --ds-seed $SEED --checkpoint-dir ../data/out/autoencoder/version_${AE_VERSION}/checkpoints

./bin/run_detection_train.py --seed $SEED --ds-seed $SEED --train-file ../data/datasets/train_ds_${SEED}.npz --val-file ../data/datasets/val_ds_${SEED}.npz

./bin/run_binary_train.py --seed $SEED --ds-seed $SEED --train-file ../data/datasets/train_ds_${SEED}.npz --val-file ../data/datasets/val_ds_${SEED}.npz

./bin/run_deblender_train.py --seed $SEED --ds-seed $SEED --ae-model-path ../models/autoencoder_${SEED}_${SEED}.pt --train-file ../data/datasets/train_ds_${SEED}.npz --val-file ../data/datasets/val_ds_${SEED}.npz
