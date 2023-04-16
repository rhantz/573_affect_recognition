"""
Driver script

CLI usage to train and run inference with w2v:
`python driver.py --train_input_data_path ../data/eng/train/eng_train_new.tsv --dev_input_data_path ../data/eng/dev/messages_dev_features_ready_for_WS_2022.tsv --dev_gold_path ../data/eng/dev/goldstandard_dev_2022.tsv --vector_type w2v --classifier svm --train --inf --output ../outputs`

Note - This script doesn't yet handle all possible options.
Current options:
W2V_Embeddings / SVM / Train with the training data and run inference on the dev data.

"""

import os
import sys
import argparse

import preprocess
from w2v_embeddings import W2V_Embeddings, Pretrained_Embeddings
import svm_model
import evaluation


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

    parser.add_argument(
        "--dev_gold_path",
        type=str,
        required=False,
        help="string path to dev gold standard from root directory",
    )

    parser.add_argument(
        "--vector_type",
        choices=["emo_bow", "w2v"],
        type=str,
        required=True,
        help="vectors module to use",
    )

    parser.add_argument(
        "--classifier",
        choices=["svm"],
        type=str,
        required=True,
        help="classifier module to use",
    )

    parser.add_argument(
        "--train", required=False, action="store_true", help="run training",
    )

    parser.add_argument(
        "--inf", required=False, action="store_true", help="run inference",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="output directory",
    )

    return parser.parse_args(sys.argv[1:])


# process data
arguments = get_args()
all_datasets = preprocess.get_datasets(arguments)
updated_datasets = preprocess.update_data(all_datasets)

# get train vectors
if arguments.train:
    essays_train = updated_datasets[0]["essay_processed"].values.tolist()
    w2v = W2V_Embeddings(essays_train)

# get dev vectors
if arguments.inf:
    essays_inf = updated_datasets[1]["essay_processed"].values.tolist()


x_train = []
y_train = updated_datasets[0]["emotion"].values.tolist()
x_inf = []

if arguments.vector_type == "w2v":
    for essay in essays_train:
        centroid = w2v.find_centroid(essay)
        x_train.append(centroid)
    for essay in essays_inf:
        centroid = w2v.find_centroid(essay)
        x_inf.append(centroid)

# add your vector type here


# choose algorithm and train model
if arguments.classifier == "svm":
    model = svm_model.get_model(x_train, y_train)

# TODO: save model

# inference

if arguments.classifier == "svm":
    predictions = svm_model.get_predictions(model, x_inf)

# save results
# TODO: add timestamp to output filename?
if arguments.inf:
    # need this because of weirdness when using arg directly with os.path.join
    output_path = arguments.output
    with open(
        os.path.join(output_path, "predictions_EMO.tsv"), "w", encoding="utf8"
    ) as f:
        for entry in predictions:
            f.write(entry + "\n")

# eval script
if arguments.inf:
    ref_path = arguments.dev_gold_path
    res_path = os.path.join(output_path, "predictions_EMO.tsv")
    output_dir = output_path
    evaluation.score(ref_path, res_path, output_dir)
