#!/bin/sh

source ~/anaconda3/etc/profile.d/conda.sh
conda activate moodmasters-env
python roberta_urdu_train.py
