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
conda env create -f moodmasters-env.yml
```

### To run the best D2 system (Bag of Words embeddings + Decision Tree classifier)...

on condor:

```
# in the top-level directory 573-affect-recognition
condor_submit D2.cmd
```

without condor:

```
# in the top-level directory 573-affect-recognition
./run.sh --config ../config/D2/D2_best_system_bow_dt.yml
```

Predictions will print to 

```
573_affect_recognition/outputs/D2/D2_best_system_bow_dt_predictions.tsv
```

Evaluation results (Scores for Macro/Micro F1, Recall, Precision and Accuracy) will print to 

```
573_affect_recognition/results/D2_scores.out
```

Best Results will be:

```
Micro Recall: 0.3407
Micro Precision: 0.3407
Micro F1-Score: 0.3407
Macro Recall: 0.2596
Macro Precision: 0.2432
Macro F1-Score: 0.2436
Accuracy: 0.3407
```

### To run other (lower-performing systems)...

1. choose a vector type from [ "bow", "emo", "emobow", "w2v", "googlePt", "glovePt" ]

2. choose a classifier from [ "dt", "svm" ]

(See end of README for glossary)

3. and run:

```
# in the top-level directory 573-affect-recognition
./run.sh --config ../config/D2/<vector_type>_<classifier>.yml
```

Predictions will print to 

```
573_affect_recognition/outputs/D2/<vector_type>_<classifier>_predictions.tsv
```

Evaluation results (Scores for Macro/Micro F1, Recall, Precision and Accuracy) will print to 

```
573_affect_recognition/results/D2_<vector_type>_<classifier>_scores.out
``` 

For example, the second best system after Bag of Words + Decision Tree (bow_dt) is Emotion Enhanced Bag of Words + Decision Tree (emobow_dt)

```
# in the top-level directory 573-affect-recognition, running:
./run.sh --config ../config/D2/emobow_dt.yml

# will produce:
573_affect_recognition/outputs/D2/emobow_dt_predictions.tsv
573_affect_recognition/results/D2_emobow_dt_scores.out
```

Second Best Results will be:

```
Micro Recall: 0.3259
Micro Precision: 0.3259
Micro F1-Score: 0.3259
Macro Recall: 0.2352
Macro Precision: 0.2154
Macro F1-Score: 0.2156
Accuracy: 0.3259
```

#### Glossary

Embeddings:

 - "bow" : Bag of Words 
 - "emo" : Emotion Lexicion only  
 - "emobow" : Emotion Enhanced Bag of Words 
 - "w2v" : Word2Vec on our training data
 - "googlePt" : Word2Vec pretrained on Google News Corpus 
 - "glovePt" : GloVe pretrained on Twitter Corpus
 
Classifiers

- "dt" : decision tree
- "svm" : SVM

