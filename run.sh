#!/bin/sh

if [ -d ~/"anaconda3" ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
elif [ -d ~/"miniconda3" ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
else
    echo "You do not seem to have conda installed"
fi

conda activate moodmasters-env
cd src
python -m spacy download en_core_web_sm
python -m nltk.downloader wordnet
python -m nltk.downloader omw
python main.py $@
