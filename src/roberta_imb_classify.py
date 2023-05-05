from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

import pandas as pd

def classify(data_list):
    '''
    Given a list of dataframe datasets, performs inference
    using distilroberta-base model that was fine-tuned on
    a version of our training data adjusted for class imbalance.
    Returns classification results as a list of strings. 
    '''

    model = AutoModelForSequenceClassification.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_imb_model")

    tokenizer = AutoTokenizer.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_imb_model")

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer)

    #data_list[1] for the dev data
    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
