import subprocess
import yaml
from pathlib import Path
import pandas as pd
from collections import defaultdict

# if there's no majority vote, tiebreak by voting for the first model in this list
# CONFIGS = ["config/D3/roberta.yml", "config/D3/roberta_imb.yml", "config/D3/emobow_bal_dt.yml"]

CONFIGS = ["../config/D4/bow_bal_dt.yml", "../config/D4/bow_bal_dtBoost.yml", "../config/D4/bow_dt.yml"]

def get_final_prediction(row_predictions):
    num_predictions = defaultdict(int)
    for prediction in row_predictions:
        num_predictions[prediction] += 1
    potential_best_prediction = max(num_predictions, key=lambda x: num_predictions[x])
    # default to first model on constant configs list
    if num_predictions[potential_best_prediction] == 1:
        return row_predictions[0]
    return potential_best_prediction

def main():
    all_predictions = pd.DataFrame()
    for item in CONFIGS:
        process_result = subprocess.run(["python", "main.py", "--config", item])
        if process_result.returncode != 0:
            raise ValueError(f"The process failed!\n{process_result.returncode}")
        with open(item, 'r') as ymlfile:
            config = yaml.load(ymlfile, Loader=yaml.Loader)
        predictions_file_path = Path(config["output_dir"])/Path(config["predictions_file"])
        predictions = pd.read_csv(predictions_file_path)
        all_predictions.append(predictions)

    final_predictions = []
    for _, row_predictions in all_predictions.iterrows():
        final_predictions.append(get_final_prediction(row_predictions))

    prediction_dataframe = pd.DataFrame({"prediction":final_predictions})
    prediction_dataframe.to_csv("D4_combined_predictions.tsv", sep='\t')


if __name__ == "__main__":
    main()