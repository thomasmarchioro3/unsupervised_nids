import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99

# Local imports
from data.cicids2017 import load_cicids2017
from data.nslkdd import load_nslkdd


def load_kddcup99(percent10: bool=False, binary: bool=True):

    data = fetch_kddcup99(percent10=percent10)
    X = data.data
    y = data.target

    feature_names = [str(feature) for feature in data.feature_names]

    X = pd.DataFrame(X, columns=feature_names)
    y = pd.Series(y).apply(lambda x: x.decode('utf-8').rstrip('.'))

    if binary:
        y = y.apply(lambda x: 0 if x == 'normal' else 1)

    return X, y


def load_dataset(dataset_name: str, use_subsample: bool=False, subsample_size=1.0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load dataset.

    Available datasets:
        - kddcup99
        - cicids2017
        - nslkdd

    Args:
        dataset_name (str): Name of the dataset.
        use_subsample (bool): Whether to use a subsample of the dataset. Default: False.
        subsample_size (float): Percentage of the dataset to use. Default: 1.0.
    """

    match dataset_name:
        case 'kddcup99':
            X, y = load_kddcup99()
        case 'cicids2017':
            X, y = load_cicids2017()
        case 'nslkdd':
            X, y = load_nslkdd()
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')

    if use_subsample:
        # subsample while preserving the row order (useful for temporal data)
        random_idx = np.random.choice(X.shape[0], size=int(X.shape[0] * subsample_size), replace=False)
        random_idx = np.sort(random_idx)
        X = X.iloc[random_idx].reset_index(drop=True)
        y = y.iloc[random_idx].reset_index(drop=True)

    # The benign traffic should always be labeled "normal"
    assert len(y == 'normal') > 0

    return X, y


if __name__ == "__main__":

    X, y = load_dataset('kddcup99', use_subsample=True, subsample_size=0.1)
    
    print(X[:5])
    print(y[:5])