#### SCORER ###
from operator import index
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, r2_score

def scorer_(data: pd.DataFrame, model, splitter: float) -> tuple:
    """Fits a given model to a given dataset and 
    returns the score on both training and testing as a tuple."""

    # Find the row to split dataset in training and testing set
    split = int(data.shape[0] * splitter)

    # Sclice the dataframe accordingly
    features_train = data.iloc[1:split,:-1]
    labels_train = data.iloc[1:split,-1]

    features_test = data.iloc[split:,:-1]
    labels_test = data.iloc[split:,-1]

    # Fit the model onto training data
    model.fit(features_train, labels_train)

    # Compute score for both training and testing
    training_score = model.score(features_train, labels_train)
    test_score = model.score(features_test, labels_test)

    return (training_score, test_score)


def evaluate_model(model, subtask, X_train, y_train, X_val, y_val) -> Dict[str, Tuple]:
    """Evaluates performance of a model wrt a specific label column"""

    model.fit(X_train, y_train)        
    
    if subtask == 1 or subtask == 2: 
        scorer = roc_auc_score

        # Get index corresponding to class 1 
        index = list(model.classes_).index(1.0)
        y_pred = model.predict_proba(X_train)[:, index]
        y_val_pred = model.predict_proba(X_val)[:, index]

    elif subtask == 3: 
        scorer = r2_score

        y_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

    training_score = scorer(y_train, y_pred)
    val_score = scorer(y_val, y_val_pred)

    return model, (training_score, val_score)


def scoreboard(models: list, subtask: int, X_train, y_trains: List[pd.Series], X_val, y_vals: List[pd.Series]) -> dict:
    """Takes a list of several models and runs them on the same dataset.
    Trains each model wrt to each label column in the list of label columns. 
    Training and Test scores are written to a dictionaru per model for each label column."""

    # Initializing output dict
    scores = dict()
    scores_model = dict()
    best_score = {y_train.name: 0 for y_train in y_trains}
    progress = dict() 

    # Train each model on each label column
    for model_class, kwargs in models:
        model = model_class(**kwargs)

        model_name = type(model).__name__
        print(f"Evaluating {model_name}...")

        scores_model = dict()
        for y_train, y_val in zip(y_trains, y_vals):

            label = y_train.name
            prev_best_score = best_score[label]
            
            model_trained, score = evaluate_model(model, subtask, X_train, y_train, X_val, y_val)
            scores_model[label] = (model_trained, score)
            
            # Printing progress
            val_score = score[1]
            if val_score > prev_best_score:
                print(f"[Progress] Best validation score on column {label}:")
                print(f"    {model_name}: {val_score:.4f}")

                best_score[label] = val_score

                if "DummyClassifier" in list(scores.keys()):
                    val_score_dummy = scores["DummyClassifier"][label][1][1]
                    print(f"    DummyClassifier: {val_score_dummy:.4f}\n")
                    print(f"    Total progress: {val_score - val_score_dummy:.4f}\n")

                    progress[label] = val_score - val_score_dummy
        
        scores[model_name] = scores_model.copy()

    print(f"Total progress: {progress}")
    return scores