#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

echo >> log.txt
cmd="./bin/run_detection_train.py --seed 41 --train-file ../data/datasets/train_ds_41_20240927143647.pt --val-file ../data/datasets/val_ds_41_20240927143647.pt"
echo $cmd >> log.txt
eval $cmd
