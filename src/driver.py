"""
Driver script
Usage note here
"""

import sys
import argparse
import pandas
import numpy as np

from sklearn import svm

from w2v_embeddings import W2V_Embeddings, Pretrained_Embeddings

# from emo_bow import EmoBoW
import preprocess


def get_args():
    """
    Returns: CLI arguments
    """

    parser = argparse.ArgumentParser(description="Reads in data")

    parser.add_argument(
        "--train_input_data_path",
        type=str,
        required=True,
        help="string path to initial training data from root directory",
    )
    parser.add_argument(
        "--dev_input_data_path",
        type=str,
        required=False,
        help="string path to initial development data from root directory",
    )
    parser.add_argument(
        "--test_input_data_path",
        type=str,
        required=False,
        help="string path to initial test data from root directory",
    )

    return parser.parse_args(sys.argv[1:])


# process data
arguments = get_args()
all_datasets = preprocess.get_datasets(arguments)
updated_datasets = preprocess.update_data(all_datasets)

# updated_datasets[0].to_clipboard() !!!

# get vectors
# updated_datasets[0] is the training data
processed_essays = updated_datasets[0]["essay_processed"].values.tolist()
w2v = W2V_Embeddings(processed_essays)


# return list of lists x and y
# here's where we get the vectors for each essay
# is there a better way of doing this than iterating?
x_list = []
for essay in processed_essays:
    centroid = w2v.find_centroid(essay)
    x_list.append(centroid)

y_list = updated_datasets[0]["emotion"].values.tolist()


# emo vectors want vocabulary


# train model

clf = svm.SVC()
clf.fit(x_list, y_list)

guess = clf.predict([x_list[0]])
print(guess)

# save model

# inference


# eval script

