from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

def classify(data_list):

    model = AutoModelForSequenceClassification.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_imb_model")

    tokenizer = AutoTokenizer.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_imb_model")


    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer)

    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
