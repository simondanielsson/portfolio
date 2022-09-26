from typing import List

import pandas as pd 

from functions.single_impute import single_impute
from functions.multiple_impute import multivariate_impute

# Threshold at 0.7
# DO NOT USE
THRESHOLD_0_7 = (
        "data/X_train_imputed.csv", 
        "data/X_val_imputed.csv",
        "data/X_test_imputed.csv",
        "data/y_train.csv",
        "data/y_val.csv"
)

# Only single imputed
THRESHOLD_1 = (
        "data/X_train_imputed_1 2.csv", 
        "data/X_val_imputed_1 2.csv",
        "data/X_test_imputed_1 2.csv",
        "data/y_train_1 2.csv",
        "data/y_val_1 2.csv"
)

# 0.95 threshold, almost only single imputed 
THRESHOLD_0_95 = (
        "data/X_train_imputed_08.csv", 
        "data/X_val_imputed_08.csv",
        "data/X_test_imputed_08.csv",
        "data/y_train.csv",
        "data/y_val.csv"
)

# Joschi's dataset
THRESHOLD_MINMAX_1 = (
        "data/X_train_imputed_minmax.csv", 
        "data/X_val_imputed_minmax.csv",
        "data/X_test_imputed_minmax.csv",
        "data/y_train.csv",
        "data/y_val.csv"
)

THRESHOLD_MINMAX_0 = (
        "data/X_train_imputed_minmax0.csv", 
        "data/X_val_imputed_minmax0.csv",
        "data/X_test_imputed_minmax0.csv",
        "data/y_train.csv",
        "data/y_val.csv"
)

PATHS = THRESHOLD_0_95


def impute_new_df(df: pd.DataFrame, single_imputers: dict, multi_imputer) -> pd.DataFrame:
    """Impute df using given imputers""" 
    df = df.copy()

    print("Imputing new df")
    # Single impute columns
    for feature, imputer in single_imputers.items():
        df[feature] = imputer.transform(df[feature].values.reshape(-1, 1))

    # Multiple impute the remaining columns
    if multi_imputer is not None:
        df = pd.DataFrame(multi_imputer.transform(df))

    return df 


def impute_single_df(df: pd.DataFrame, threshold: float, max_iter: int, verbose: int = 0) -> pd.DataFrame:
    """Perform single imputation if proportion of missing values of a column is 
    less than threshold, and performs multivariate imputation otherwise.
    
    Note that mutlivariate imputation takes very long to converge; you'll likely want to 
    perform it once and save it to disk.
    
    If threshold == 1 then multi_imputer is None"""

    df_single, single_imputers = single_impute(df, threshold) 
    if threshold == 1:
        df_multi = df_single
        multi_imputer = None
    else:
        df_multi, multi_imputer = multivariate_impute(df_single, max_iter=max_iter, verbose=verbose)

    # You'll need the imptuter objects when imputing any new data, e.g. the test set 
    return df_multi, (single_imputers, multi_imputer)


def impute(X_train, X_val, X_test, y_train, y_val, threshold, max_iter = 100, verbose = 0) -> None:
    """Impute data on basis of training set and save as csv"""   

    X_train, (single_imputers, multi_imputer) = impute_single_df(X_train, threshold, max_iter, verbose)

    X_val, X_test = (
        impute_new_df(X_val, single_imputers, multi_imputer), 
        impute_new_df(X_test, single_imputers, multi_imputer)
    )

    for path, data in zip(PATHS, [X_train, X_val, X_test, y_train, y_val]):
        if verbose != 0:
            print(f"Saving data to {path}")
        data.to_csv(path, header=data.columns)



def load_imputed_data() -> List[pd.DataFrame]:
    data = [pd.read_csv(path, index_col=0) for path in PATHS]

    return data



"""
SUBTASK 1.
Results on 0.95 threshold dataset:

DummyClassifier(): (0.5804753309265944, 0.5786980171959993),
AdaBoostClassifier(random_state=1): (0.6981046931407943, 0.6739778908580453),
LinearRegression(): (0.08428222112626971, -11704924773.964266),
LassoCV(random_state=1): (0.06783578519022548, 0.04429180435277791),
GridSearchCV(RandomForestClassifier(),
    param_grid={'min_samples_split': [2, 5, 10, 50, 100]}): (0.9998495788206979, 0.6818740129847342),
XGBRFClassifier(...): (0.7002105896510229, 0.6683628706790665)}


Depending on the results in the other tasks this might actually be sufficient performance 
"""