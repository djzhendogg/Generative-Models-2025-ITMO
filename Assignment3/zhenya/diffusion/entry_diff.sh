#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=$(pwd)
python3 train_diff.py > diff_log_zenya.log