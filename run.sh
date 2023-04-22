#!/bin/sh
source ~/anaconda3/etc/profile.d/conda.sh
conda activate moodmasters-env
cd src
python main.py $@
