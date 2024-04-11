#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="5"

for i in {42..44};
do
    cmd="./bin/run_detection_train_script.py -s $i --star-density 0 -o -t "11_${i}""
    echo $cmd > log.txt
    eval $cmd
done
