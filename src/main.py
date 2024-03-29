"""
main script
CLI usage to train and run inference:

`python main.py
--train_input_data_path ../data/eng/train/eng_train_new.tsv
--dev_input_data_path ../data/eng/dev/eng_dev_combined.tsv
--dev_gold_path ../data/eng/dev/goldstandard_dev_2022.tsv
--vector_type : one of ["emo_bow", "w2v", "pretrained", "bow_only", "emo_only"]
--pretrained_model : one of ["word2vec-google-news-300", "glove-twitter-25"]
--classifier : one of ['svm', 'svmboost', 'dt', 'dtboost', 'roberta', 'roberta_imb','roberta_urdu','roberta_urdu_imb']
--train
--inf
--negation
--output_dir ../outputs/D2
--predictions_file predictions_EMO.tsv
--results_dir ../results
--score_file D2_scores.out`

or

`python main.py
--config ../config/<deliverable>/<config_name>.yml
"""

import os
import sys
import argparse
import yaml

import preprocess
import create_vectors
import classify
import evaluation
import roberta_classify
import roberta_imb_classify
import roberta_urdu_classify
import roberta_urdu_imb_classify
import class_imbalance
import preprocess_urdu

def get_args():
    """
    Returns: CLI arguments
    """
    sys_args = sys.argv[1:]
    parser = argparse.ArgumentParser(description="Reads in data")

    if "--config" in sys_args:

        parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="string path to config file with required arguments",
        )

        args = parser.parse_args(sys_args)

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
            args.__dict__.update(config)
    else:

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
            choices=["svm", "svmboost", "dt", "dtboost", "roberta","roberta_imb", "roberta_urdu","roberta_urdu_imb"],
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
            "--negation", required=False, action="store_true", help="remove negation",
        )

        parser. add_argument(
            "--random_over_under_sample", action="store_true", help="apply random_over_under_sample class_imbalance method"
        )
        parser.add_argument(
            "--output_dir", type=str, required=True, help="output directory",
        )

        parser.add_argument(
            "--results_dir", type=str, required=True, help="results directory",
        )

        parser.add_argument(
            "--score_file", type=str, required=True, help="name of score file",
        )

        parser.add_argument(
            "--predictions_file", type=str, required=True, help="name of predictions file",
        )

        parser.add_argument(
            "--urdu", action="store_true", help="use urdu pathway"
            )

        args = parser.parse_args(sys_args)
    return args


def write_predictions(predictions):
    output_path = arguments.output_dir
    with open(
        os.path.join(output_path, arguments.predictions_file), "w", encoding="utf8"
    ) as f:
        for entry in predictions:
            f.write(entry + "\n")


def run_eval():
    ref_path = arguments.dev_gold_path
    output_path = arguments.output_dir
    output_path = os.path.join(output_path, arguments.predictions_file)
    results_path = arguments.results_dir
    score_name = arguments.score_file
    evaluation.score(ref_path, output_path, results_path, score_name)


if __name__ == "__main__":
    arguments = get_args()
    datasets = preprocess.get_datasets(arguments)

    try:
        if arguments.urdu:
            updated_datasets = preprocess_urdu.update_data(datasets)
    except AttributeError:
        try:
            negation = arguments.negation
            updated_datasets = preprocess.update_data(datasets, True)
        except AttributeError:
            updated_datasets = preprocess.update_data(datasets, False)

    formatted_data = create_vectors.make_vectors(arguments, updated_datasets)

    if arguments.random_over_under_sample:
        x, y = formatted_data[0]
        x_new, y_new = class_imbalance.random_over_under_sample(x, y)
        formatted_data[0] = (x_new, y_new)

    if arguments.classifier == "roberta":
        predictions = roberta_classify.classify(datasets)
    elif arguments.classifier == "roberta_imb":
        predictions = roberta_imb_classify.classify(datasets)
    elif arguments.classifier == "roberta_urdu":
        predictions = roberta_urdu_classify.classify(datasets)
    elif arguments.classifier == "roberta_urdu_imb":
        predictions = roberta_urdu_imb_classify.classify(datasets)
    else:
        predictions = classify.train_and_classify(formatted_data, arguments.classifier)
    
    write_predictions(predictions)
    run_eval()
