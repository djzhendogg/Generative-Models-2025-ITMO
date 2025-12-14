#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
export PYTHONPATH=$(pwd)
python3 train_diff.py > diff_log_zenya.log