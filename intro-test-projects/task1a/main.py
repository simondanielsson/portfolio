import pandas as pd

from sklearn.linear_model import RidgeCV

def run_pipeline(path: str) -> None:
    data = pd.read_csv(path)

    data.head()

if __name__ == "__main__":

    path = "./train.csv"
    run_pipeline(path)    
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def run_pipeline(path: str) -> None:
    data = pd.read_csv(path)
    
    y = data["y"]
    X = data[data.columns[1:]]
    
    lambdas = {"alpha": [0.1, 1, 10, 100, 200]}
    
    model = Ridge()
    
    scoring = "neg_root_mean_squared_error"
    gs = GridSearchCV(model, param_grid=lambdas, cv=10, scoring=scoring)
    
    gs.fit(X, y)
    
    RMSE_avg = - gs.cv_results_["mean_test_score"]
    
    # To csv
    out_path = "./submission.csv"
    result = pd.Series(RMSE_avg)
    result.to_csv(out_path, header=False, index=False)


if __name__ == "__main__":

    path = "./train.csv"
    run_pipeline(path)    
