import pandas as pd

def csv_to_df(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
