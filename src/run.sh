#!/bin/sh
conda env create -f moodmasters-env.yml
conda activate moodmasters-env
python ./main.py $@