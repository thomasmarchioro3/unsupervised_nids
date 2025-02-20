import numpy as np
import pandas as pd

DEFAULT_FILE_PATH = 'data/cicids2017.csv'
DEFAULT_FILE_PATH_10_PERCENT = 'data/cicids2017/cicids2017_random10.csv'

NON_FEATURE_COLUMNS = ['id', 'Flow ID', 'Timestamp', 'Label']

def load_cicids2017(filepath: str=None, random_10_percent: bool=True) -> tuple[pd.DataFrame, pd.Series]:
    
    if filepath is None:
        filepath = DEFAULT_FILE_PATH_10_PERCENT if random_10_percent else DEFAULT_FILE_PATH

    df = pd.read_csv(filepath)
    X = df.drop(columns=NON_FEATURE_COLUMNS)
    y = df['Label']
    y = y.replace({'BENIGN': 'normal'})

    return X, y