# 573_affect_recognition

## Team Moodmasters

The present repository is designed to run an end-to-end multi-class emotion classification system on essays. This system is designed in respose to track 2 of the WASSA 2022 Shared Task on Empathy Detection and Emotion Classification. Click [here](https://codalab.lisn.upsaclay.fr/competitions/834) to learn more!

### To set up your environment for the system, please follow the steps described below:

1. Clone the repository in patas or locally:

```
git clone git@github.com:rhantz/573_affect_recognition.git
```

2. Reproduce the conda environment used for the project. Use this command:

```
# in the top-level directory 573_affect_recognition
conda env create -f src/moodmasters-env.yml
```

### To run the best D3 system (Roberta model + class balancing)...

on condor:

```
# in the top-level directory 573_affect_recognition
condor_submit D3.cmd
```

without condor:

```
# in the top-level directory 573_affect_recognition
./run.sh --config ../config/D3/roberta_imb.yml
```

Predictions will print to 

```
573_affect_recognition/outputs/D3/roberta_imb_predictions.tsv
```

Evaluation results (Scores for Macro/Micro F1, Recall, Precision and Accuracy) will print to 

```
573_affect_recognition/results/D3_scores.out
```

Best Results will be:

```
Micro Recall: 0.5148
Micro Precision: 0.5148
Micro F1-Score: 0.5148
Macro Recall: 0.536
Macro Precision: 0.4735
Macro F1-Score: 0.4588
Accuracy: 0.5148
```


#### Glossary

Embeddings:

 - "bow" : Bag of Words 
 - "emo" : Emotion Lexicon only  
 - "emobow" : Emotion Enhanced Bag of Words 
 - "w2v" : Word2Vec on our training data
 - "googlePt" : Word2Vec pretrained on Google News Corpus 
 - "glovePt" : GloVe pretrained on Twitter Corpus
 
Classifiers

- "dt" : decision tree
- "svm" : SVM
- "roberta" : distilroberta-base model fine-tuned on our training data
- "roberta_imb" : distilroberta-base model fine-tuned on our training data with random over/under sampling class balancing strategy

