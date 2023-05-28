# 573_affect_recognition

## Team Moodmasters

The present repository is designed to run an end-to-end multi-class emotion classification system on English essays. This PRIMARY task system is designed in respose to track 2 of the WASSA 2022 Shared Task on Empathy Detection and Emotion Classification. Click [here](https://codalab.lisn.upsaclay.fr/competitions/834) to learn more! 

We also ADAPTED this system in language and genre to perform emotion classification on Urdu tweets. This is a modified version of track A of the EmoThreat@FIRE 2022 Shared Task -- EmoThreat: Emotions & Threat Detection in Urdu. Click [here](https://sites.google.com/view/multi-label-emotionsfire-task/home) to learn more! (We only utilized data instances with single emotion labels in order to perform multi-class classification.)

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

### To run the best PRIMARY task system (Ensemble of Roberta, Roberta + Class Balancing, EmoBow Balanced Decision Tree) ...

```
# in the top-level directory 573_affect_recognition

# TEST
./ensemble_voting.sh ../config/D4/roberta_test.yml ../config/D4/roberta_imb_test.yml ../config/D4/emobow_bal_dt_test.yml test
# DEV
./ensemble_voting.sh ../config/D4/roberta_dev.yml ../config/D4/roberta_imb_dev.yml ../config/D4/emobow_bal_dt_dev.yml dev
```

Predictions will print to 

```
# TEST 
573_affect_recognition/outputs/D4/primary/evaltest/D4_combined_predictions.tsv
# DEV
573_affect_recognition/outputs/D4/primary/devtest/D4_combined_predictions.tsv
```

Evaluation results (Scores for Macro/Micro F1, Recall, Precision and Accuracy) will print to 

```
# TEST
573_affect_recognition/results/D4/primary/evaltest/D4_scores.out
# DEV
573_affect_recognition/results/D4/primary/devtest/D4_scores.out
```

Test results will be:

```
Micro Recall: 0.5446
Micro Precision: 0.5446
Micro F1-Score: 0.5446
Macro Recall: 0.3791
Macro Precision: 0.3983
Macro F1-Score: 0.3697
Accuracy: 0.5446
```

Dev results will be:

```
Micro Recall: 0.6296
Micro Precision: 0.6296
Micro F1-Score: 0.6296
Macro Recall: 0.4796
Macro Precision: 0.6862
Macro F1-Score: 0.5026
Accuracy: 0.6296
```

### To run the best ADAPTATION task system (Roberta model, no class balancing)...

on condor (TEST only):

```
# in the top-level directory 573_affect_recognition
condor_submit D4.cmd
```

without condor:

```
# in the top-level directory 573_affect_recognition

# TEST
./run.sh --config ../config/D4/roberta_urdu_test.yml
# DEV
./run.sh --config ../config/D4/roberta_urdu_dev.yml
```

Predictions will print to 

```
# TEST
573_affect_recognition/outputs/D4/adaptation/evaltest/roberta_urdu_predictions.tsv
# DEV
573_affect_recognition/outputs/D4/adaptation/devtest/roberta_urdu_predictions.tsv
```

Evaluation results (Scores for Macro/Micro F1, Recall, Precision and Accuracy) will print to 

```
# TEST
573_affect_recognition/results/D4/adaptation/evaltest/D4_scores.out
# DEV
573_affect_recognition/results/D4/adaptation/devtest/D4_scores.out
```

Test results will be:

```
Micro Recall: 0.8381
Micro Precision: 0.8381
Micro F1-Score: 0.8381
Macro Recall: 0.4851
Macro Precision: 0.6739
Macro F1-Score: 0.4843
Accuracy: 0.8381
```

Dev results will be:

```
Micro Recall: 0.8496
Micro Precision: 0.8496
Micro F1-Score: 0.8496
Macro Recall: 0.513
Macro Precision: 0.7601
Macro F1-Score: 0.5292
Accuracy: 0.8496
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
- "roberta_urdu" : urduhack/roberta-urdu-small model fine-tuned on our training data
- "roberta_urdu_imb": urduhack/roberta-urdu-small model fine-tuned on our training data with random over/under sampling class balancing strategy

