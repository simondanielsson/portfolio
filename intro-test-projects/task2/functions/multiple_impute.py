from typing import Tuple
import pandas as pd

from sklearn.linear_model import BayesianRidge

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def multivariate_impute(df: pd.DataFrame, 
                        max_iter: int = 10, 
                        estimator = BayesianRidge(),
                        verbose: int = 0) -> Tuple[pd.DataFrame, IterativeImputer]:

    print("Multiple imputing...")
    imputer = IterativeImputer(estimator=estimator, max_iter=max_iter, verbose=verbose)
    df_imputed = pd.DataFrame(imputer.fit_transform(df))
    
    print("Finished multiple imputing...")
    return df_imputed, imputer