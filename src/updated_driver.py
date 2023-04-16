"""
Driver script
CLI usage to train and run inference with w2v:
`python driver.py
--train_input_data_path ../data/eng/train/eng_train_new.tsv
--dev_input_data_path ../data/eng/dev/dev/combined_dev_data.tsv
--dev_gold_path ../data/eng/dev/goldstandard_dev_2022.tsv
--vector_type : one of ["emo_bow", "w2v", "pretrained", "bow_only", "emo_only"]
--pretrained_model : one of ["word2vec-google-news-300", "glove-twitter-25"]
--classifier : one of ['svm', 'dt']
--train
--inf
--output ../outputs`

"""

import os
import sys
import argparse

import preprocess
import create_vectors
import classify
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
        "--dev_gold_path",
        type=str,
        required=False,
        help="string path to dev gold standard from root directory",
    )

    parser.add_argument(
        "--test_input_data_path",
        type=str,
        required=False,
        help="string path to initial test data from root directory",
    )

    parser.add_argument(
        "--vector_type",
        choices=["emo_bow", "w2v", "pretrained", "bow_only", "emo_only"],
        type=str,
        required=True,
        help="vectors module to use",
    )

    parser.add_argument(
        "--pretrained_model",
        choices=["word2vec-google-news-300", "glove-twitter-25"],
        type=str,
        required=False,
        help="name of pretrained model",
    )

    parser.add_argument(
        "--classifier",
        choices=["svm", "dt"],
        type=str,
        required=True,
        help="classifier module to use",
    )

    parser.add_argument(
        "--train",
        required=False,
        action="store_true",
        help="run training",
    )

    parser.add_argument(
        "--inf",
        required=False,
        action="store_true",
        help="run inference",
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output directory",
    )

    return parser.parse_args(sys.argv[1:])

def write_predictions(predictions):
    output_path = arguments.output
    with open(
            os.path.join(output_path, "predictions_EMO.tsv"), "w", encoding="utf8"
    ) as f:
        for entry in predictions:
            f.write(entry + "\n")

def run_eval():
    output_path = arguments.output
    ref_path = arguments.dev_gold_path
    res_path = os.path.join(output_path, "predictions_EMO.tsv")
    output_dir = output_path
    evaluation.score(ref_path, res_path, output_dir)

if __name__ == '__main__':
    arguments = get_args()
    datasets = preprocess.get_datasets(arguments)
    updated_datasets = preprocess.update_data(datasets)
    formatted_data = create_vectors.make_vectors(arguments, updated_datasets)
    predictions = classify.train_and_classify(formatted_data, arguments.classifier)
    write_predictions(predictions)
    run_eval()