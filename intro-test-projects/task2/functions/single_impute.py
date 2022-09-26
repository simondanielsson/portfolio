from typing import Tuple

import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer

def _single_impute_col(df: pd.DataFrame, feature_name: list, strategy: str = "mean") -> Tuple[pd.DataFrame, object]:
    """Single impute a single column feature_name of df"""
    imputer = SimpleImputer(strategy=strategy)
    
    col = df.loc[:, feature_name]
    df.loc[:, feature_name] = imputer.fit_transform(col.values.reshape(-1, 1))
    
    return df, imputer
    
    
def _perform_single_impute(df: pd.DataFrame, features_to_impute: list) -> Tuple[pd.DataFrame, dict]:
    """Single impute all columns of df in features_to_import"""
    df_ts = df.copy()
    imputers = dict()

    # Simple impute over each time series column (i.e. across different patients)
    for feature_name in features_to_impute:
        df_ts, imputer = _single_impute_col(df_ts, feature_name)

        imputers[feature_name] = imputer
            
    return df_ts, imputers


def _prop_missing(df: pd.DataFrame) -> dict:
    """Compute proportion of missing values per column of df"""
    prop_missing_vals = {}
    tot_rows = df.shape[0]
    
    for col_name in df.columns:
        missing_vals = np.sum(df.loc[:, col_name].isna())
        prop_missing_vals[col_name] = missing_vals / df.shape[0]
        
    return prop_missing_vals


def _find_impute_features(df: pd.DataFrame, missing_threshold: float) -> list:
    """Find features to single impute depending on threshold value"""
    missing = _prop_missing(df)
    features_to_impute = []
    
    for feature, val in missing.items():
        if val <= missing_threshold:
            features_to_impute += [feature]
        
    print("Single imputing", 100*len(features_to_impute)/df.shape[1], "% of the columns")
    return features_to_impute


def single_impute(df: pd.DataFrame, missing_val_threshold: float) -> Tuple[pd.DataFrame, dict]:
    """Single impute columms of df for which less than missing_val_threshold of rows have missing values"""
    print("Single imputing...")
    features_to_impute = _find_impute_features(df, missing_val_threshold)
    df_imputed, imputers = _perform_single_impute(df, features_to_impute)      
    
    return df_imputed, imputers
