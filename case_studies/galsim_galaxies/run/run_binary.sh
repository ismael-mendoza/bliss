#!/usr/bin/env bash

echo >> log.txt
cmd="./bin/run_binary_train_script.py -s 42 -t "12_42""
echo $cmd >> log.txt
eval $cmd
