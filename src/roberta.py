from transformers import pipeline

import pandas as pd


def classify(data_list):
    classifier = pipeline(
        "text-classification", model="j-hartmann/emotion-english-distilroberta-base"
    )

    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
