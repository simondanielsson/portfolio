from typing import List
import argparse

from random import random
import pandas as pd
from pprint import pprint

from functions.scoring import scoreboard
from functions.select_feature import select_feature, select_labels
from functions.impute import impute, load_imputed_data
from functions.preprocess import preprocess 
from functions.models import get_models 
from functions.get_best_models import get_best_models
from functions.predict_labels import predict_labels

from sklearn.model_selection import train_test_split

def save_predictions(labels_pred, subtask):
    """Save predictions to csv for a specific subtask to some path"""
    pass 


def load_data(paths: List[str]) -> List[pd.DataFrame]:
    """Loads data from paths and returns the data as a list of dataframes"""
    dataframes = []

    for path in paths:
        dataframes.append(pd.read_csv(path, index_col=0))

    return dataframes


def present_results(scores: dict) -> None: 
    """Presents scores"""    
    # TODO: implement something more verbose, graphs etc. 
    pprint(scores)


def main(in_paths: str, subtask: int, threshold: float = None, verbose: int = 0) -> None:
    """Evaluate models on data set
    in_path: path of original data
    If update, then imputation is re-done and saved as csv. Otherwise, load imputed data from csv"""
    random_state = 1

    update = True if threshold is not None else False 
    
    if update:
        # Load train and test data
        X, y, X_test = load_data(in_paths)

        # Split data into train and validation set
        train_size = 0.7 # TODO: set

        X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=train_size, random_state=random_state)
    
        # Impute and load data
        max_iter = 1
        
        impute(
            X_train, 
            X_val, 
            X_test, 
            y_train, 
            y_val, 
            threshold=threshold, 
            max_iter=max_iter, 
            verbose=verbose
        )

    X_train, X_val, X_test, y_train, y_val = load_imputed_data()

    # Select features
    """
    print("Selecting features...")
    feature_indeces_to_select = select_feature(X_train, y_train)
    X_train, X_val, X_test= (
        X_train.iloc[:,feature_indeces_to_select],
        X_val.iloc[:,feature_indeces_to_select],
        X_test.iloc[:,feature_indeces_to_select],
    )
    """

    # Select label features for this specific problem 
    print("Selecting labels...")
    y_trains, y_vals = select_labels(subtask, y_train, y_val)

    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    
    # Evaluate models
    print("Fetching models...")
    models = get_models(subtask)

    # Evaluate models
    print(f"Evaluating models...")
    scores = scoreboard(models, subtask, X_train, y_trains, X_val, y_vals)

    # Present scores
    present_results(scores)

    # Fetch best model for each label column
    best_models = get_best_models(scores, subtask)

    # Make probabilistic predictions on test data
    labels_pred = predict_labels(best_models, subtask, X_test)

    # Save predictions to csv
    save_predictions(labels_pred, subtask)
    
    
    

    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run model pipeline for subtask 1')
    parser.add_argument("subtask", metavar="N", type=int, nargs=1, help="Choice of subtask to be run")
    parser.add_argument("-u", "--update-imputation", action="store", metavar="T", help="Update the imputated dataset using threshold T, and save as csv")
    parser.add_argument("-v", "--verbose", action="store_const", const=2, help="Verbose output")
    args = vars(parser.parse_args())

    # Initialize 
    subtask = args["subtask"][0]
    verbose = args["verbose"] if not "None" else 0

    # Update imputation: None if no update, else threshold value  
    t = args["update_imputation"]
    threshold = float(t) if t is not None else None 

    # Paths for loading original data used for imputation
    #in_paths =["data/train_features_ts 2.csv", "data/train_labels.csv", "data/test_features_ts 2.csv"]
    in_paths =["data/train_features_minmax.csv", "data/train_labels.csv", "data/test_features_minmax.csv"]
    main(in_paths, subtask=subtask, threshold=threshold, verbose=verbose)





# Comments:
"""
Might want to try to do forward-fill imputation per patient if Joschi's dataset does not work out


"""