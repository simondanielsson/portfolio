from sklearn.feature_selection import SelectFromModel 
from sklearn . ensemble import RandomForestRegressor 
import pandas as pd
import numpy as np

def select_feature(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Select the most important features in X_train based on importance in RandomForestRegressor.
    """

    sfm = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=38) , threshold='median' )
    sfm.fit(X_train, y_train) 
    X_train_sfm = sfm.transform(X_train) 

    mask_sfm = sfm.get_support()

    X_train_selected = X_train.iloc[:,mask_sfm]
    
    return X_train_selected