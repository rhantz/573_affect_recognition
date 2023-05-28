import subprocess
import yaml
from pathlib import Path
import pandas as pd
from collections import defaultdict
import csv
import evaluation
import sys
import argparse
import os

# if there's no majority vote, tiebreak by voting for the first model in this list

# roberta, roberta_imb, emo_bow_bal_dt
CONFIGS = [sys.argv[1], sys.argv[2], sys.argv[3]]
data = sys.argv[4]
# CONFIGS = ["../config/D3/roberta_dev.yml", "../config/D3/roberta_imb_dev.yml", "config/D3/emobow_bal_dt_dev.yml"]

#for local testing
# CONFIGS = ["../config/D4/bow_bal_dt.yml", "../config/D4/bow_bal_dtBoost.yml", "../config/D4/bow_dt.yml"]


def main():
    list_of_prediction_lists = []
    for idx, item in enumerate(CONFIGS):
        process_result = subprocess.run(["python", "main.py", "--config", item])
        if process_result.returncode != 0:
            raise ValueError(f"The process failed!\n{process_result.returncode}")
        with open(item, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.Loader)
        predictions_file_path = Path(f"{config['output_dir']}/{config['predictions_file']}")
        prediction_list = []
        with open(predictions_file_path, newline='') as csvfile:
            prediction_reader = csv.reader(csvfile, delimiter=' ')
            for row in prediction_reader:
                prediction_list.append((', '.join(row)))
        list_of_prediction_lists.append(prediction_list)


    final_predictions = []
    for pred_1, pred_2, pred_3 in zip(list_of_prediction_lists[0], list_of_prediction_lists[1], list_of_prediction_lists[2]):
        # Create a dictionary to store the counts of each unique prediction
        counts = {}
        # Count the occurrences of each prediction
        for pred in [pred_1, pred_2, pred_3]:
            if pred in counts:
                counts[pred] += 1
            else:
                counts[pred] = 1
        # Find the prediction that occurs at least twice
        majority_prediction = None
        for pred, count in counts.items():
            if count >= 2:
                majority_prediction = pred
                break
        # If no prediction occurs at least twice, take the value from the first list
        if majority_prediction is None:
            majority_prediction = pred_1
        # Add the majority prediction to the final predictions list
        final_predictions.append(majority_prediction)

    prediction_dataframe = pd.DataFrame(final_predictions)
    prediction_dataframe = prediction_dataframe.reset_index(drop=True)
    if data == 'dev':
    # print to d4/primary/devtest, run eval
        prediction_dataframe.to_csv("../outputs/D4/primary/devtest/D4_combined_predictions.tsv", sep='\t', header=0, index=False)
        ref_path = '../data/eng/dev/goldstandard_dev_2022.tsv'
        output_path = '../outputs/D4/primary/devtest/D4_combined_predictions.tsv'
        results_path = '../results/D4/primary/devtest'
        score_name = 'D4_main_task_dev_results.out'
        evaluation.score(ref_path, output_path, results_path, score_name)
    elif data == 'test':
    # print to d4/primary/evaltest, run eval
        prediction_dataframe.to_csv("../outputs/D4/primary/evaltest/D4_combined_predictions.tsv", sep='\t', header=0, index=False)
        ref_path = "../data/eng/test/eng_test_golds.tsv"
        output_path = '../outputs/D4/primary/evaltest/D4_combined_predictions.tsv'
        results_path = '../results/D4/primary/evaltest'
        score_name = 'D4_main_task_test_results.out'
        evaluation.score(ref_path, output_path, results_path, score_name)
    else:
        raise ValueError("You did not choose test or dev.")

if __name__ == "__main__":
    main()