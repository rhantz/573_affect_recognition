from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import evaluate

import numpy as np
import pandas as pd

import class_imbalance

''' 
Fine-tunes the vanilla distilroberta-base model from huggingface
on our training data and saves the resulting model to patas directory.
Note that the training line at bottom is commented out - uncomment to train.
May be run on condor by submitting condor_roberta_train.cmd
'''

#Define metric for transformers package to evaluate training progress
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels,average="macro")

#Tokenizer function
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

#Instantiate tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
model = AutoModelForSequenceClassification.from_pretrained("distilroberta-base", num_labels=7)

#Load metric
metric = evaluate.load("f1")

#Indicate training and dev files. Files are loaded as dataframes
#mainly for historical reasons.
train_file = "../data/eng/train/eng_train_new.tsv"
train_data = pd.read_table(train_file, header=0)
dev_file = "../data/eng/dev/eng_dev_combined.tsv"
dev_data = pd.read_table(dev_file, header=0)
list_of_df = [train_data,dev_data]

#Get only relevant columns from training and dev data 
# and rename to `test` and `labels`
training_data = list_of_df[0]
training_data = training_data[["essay","emotion"]]
training_data = training_data.rename(columns={"essay": "text","emotion":"labels"})

dev_data = list_of_df[1]
dev_data = dev_data[["essay","emotion"]]
dev_data = dev_data.rename(columns={"essay": "text","emotion":"labels"})

#class imbalance averaging below - Comment this section out to 
# fine-tune on original training data without class imbalance averaging

x = training_data["text"].tolist()
x = np.array(x).reshape(-1,1)
y = training_data["labels"].tolist()
x_new, y_new = class_imbalance.random_over_under_sample(x, y)
x_new = x_new.flatten()
training_data = pd.DataFrame({"text":x_new, "labels":y_new})

#end class imbalance averaging

#Convert to huggingface Datasets
train_dataset = Dataset.from_pandas(training_data,split="train")
train_dataset = train_dataset.class_encode_column("labels")
dev_dataset = Dataset.from_pandas(dev_data,split="test")
dev_dataset = dev_dataset.class_encode_column("labels")

#Tokenize datasets
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_dev_dataset = dev_dataset.map(tokenize_function, batched=True)

#Specify training arguments
#Used default hyperparameters from huggingface tutorial
training_args = TrainingArguments(
    output_dir="roberta_imb_output",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    push_to_hub=False,
)

#Instantiate trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_dev_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

#uncomment here to train model!
#trainer.train()

#save model
pt_save_directory = "/home2/esokada/LING573/573_affect_recognition/src/roberta_imb_model"
tokenizer.save_pretrained(pt_save_directory)
model.save_pretrained(pt_save_directory)