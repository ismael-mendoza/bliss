#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="2"

./run_detection.sh
./run_binary.sh
./run_deblender.sh
