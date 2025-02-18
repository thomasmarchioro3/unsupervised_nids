import numpy as np
import pandas as pd

from sklearn.datasets import fetch_kddcup99


def load_kddcup99():

    data = fetch_kddcup99(percent10=True)
    X = data.data
    y = data.target

    X = pd.DataFrame(X, columns=data.feature_names)
    y = pd.Series(y).apply(lambda x: x.decode('utf-8').rstrip('.'))

    return X, y

def load_dataset(dataset_name: str, use_subsample: bool=False, subsample_size=1.0):
    """
        Load dataset.

        Available datasets:
            - kddcup99

        Args:
            dataset_name (str): Name of the dataset.
            use_subsample (bool): Whether to use a subsample of the dataset. Default: False.
            subsample_size (float): Percentage of the dataset to use. Default: 1.0.
    """

    match dataset_name:
        case 'kddcup99':
            X, y = load_kddcup99()
        case _:
            raise ValueError(f'Unknown dataset: {dataset_name}')

    if use_subsample:
        # subsample while preserving the row order (useful for temporal data)
        random_idx = np.random.choice(X.shape[0], size=int(X.shape[0] * subsample_size), replace=False)
        random_idx = np.sort(random_idx)
        X = X.iloc[random_idx].reset_index(drop=True)
        y = y.iloc[random_idx].reset_index(drop=True)

    return X, y


if __name__ == "__main__":

    X, y = load_dataset('kddcup99', use_subsample=True, subsample_size=0.1)
    
    print(X[:5])
    print(y[:5])