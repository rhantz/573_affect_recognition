from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

import pandas as pd

def classify(data_list):
    '''
    Given a list of dataframe datasets, performs inference
    using roberta-urdu-small as is and not fine-tuned.
    Returns classification results as a list of strings. 
    '''
    model = AutoModelForSequenceClassification.from_pretrained("/home2/hantz1rk/ling573/573_affect_recognition/src/roberta_urdu_output/checkpoint-342")

    tokenizer = AutoTokenizer.from_pretrained("/home2/hantz1rk/ling573/573_affect_recognition/src/roberta_urdu_output/checkpoint-342") 

    classifier = pipeline(
        "text-classification", model=model, tokenizer=tokenizer, max_length=512, truncation=True) 

    #data_list[1] for the dev data
    bare_essays = list(data_list[1]["essay"])
    output = classifier(bare_essays)

    result = []
    for entry in output:
        result.append(entry["label"])
    return result
