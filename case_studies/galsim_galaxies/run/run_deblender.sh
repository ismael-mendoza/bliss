#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

echo >> log.txt
cmd="./bin/run_deblender_train.py -s 43 -t "15_42""
echo $cmd >> log.txt
eval $cmd
