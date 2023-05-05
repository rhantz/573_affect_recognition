from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

def classify(data_list):
    '''
    Given a list of dataframe datasets, performs inference
    using distilroberta-base model fine-tuned on our training data.
    Returns classification results as a list of strings. 
    '''

    model = AutoModelForSequenceClassification.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_output/checkpoint-206")

    tokenizer = AutoTokenizer.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_output/checkpoint-206")

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer)

    #data_list[1] for the dev data
    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
