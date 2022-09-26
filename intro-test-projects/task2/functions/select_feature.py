from typing import List

from sklearn.feature_selection import SelectFromModel 
from sklearn.ensemble import RandomForestRegressor 
import pandas as pd
import numpy as np

def select_feature(X_train: pd.DataFrame, y_train: pd.DataFrame) -> list:
    """
    Select the most important features in X_train based on importance in RandomForestRegressor. 
    Returns list of indeces of features to use. 
    """

    sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=38) , threshold='median' )
    sfm.fit(X_train, y_train) 
    #X_train_sfm = sfm.transform(X_train) 

    mask_sfm = sfm.get_support(indeces=True)

    #X_train_selected = X_train.iloc[:,mask_sfm]
    
    return mask_sfm

def select_labels(subtask: int, y_train: pd.DataFrame, y_val: pd.DataFrame) -> List[pd.Series]:
    """Selects the label columns of interest and returns them as a list of series"""

    LABELS = {
        "1": "LABEL_BaseExcess, LABEL_Fibrinogen, LABEL_AST, LABEL_Alkalinephos, LABEL_Bilirubin_total, LABEL_Lactate, LABEL_TroponinI, LABEL_SaO2, LABEL_Bilirubin_direct, LABEL_EtCO2".split(", "),
        "2": "LABEL_Sepsis",
        "3": "LABEL_RRate, LABEL_ABPm, LABEL_SpO2, LABEL_Heartrate".split(", ") 
    }

    labels = LABELS[str(subtask)]
    
    y_trains = [y_train.loc[:, label] for label in labels]
    y_vals = [y_val.loc[:, label] for label in labels]

    return y_trains, y_vals 
