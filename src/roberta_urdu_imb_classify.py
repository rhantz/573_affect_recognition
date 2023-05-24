from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd

def classify(data_list):
    '''
    Given a list of dataframe datasets, performs inference
    using roberta_urdu_imb.
    Returns classification results as a list of strings. 
    '''
    model = AutoModelForSequenceClassification.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_urdu_imb_model")

    tokenizer = AutoTokenizer.from_pretrained("/home2/esokada/LING573/573_affect_recognition/src/roberta_urdu_imb_model") 

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, max_length=512, truncation=True) 

    #data_list[1] for the dev data
    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
