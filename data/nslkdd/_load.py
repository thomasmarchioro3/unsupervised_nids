import pandas as pd

from ._descriptor import COLUMN_NAMES


DEFAULT_TRAIN_FILE_PATH = "data/nslkdd/KDDTrain+.txt"
DEFAULT_TRAIN_FILE_PATH_20_PERCENT = "data/nslkdd/KDDTrain+_20Percent.txt"
DEFAULT_TEST_FILE_PATH = "data/nslkdd/KDDTest+.txt"
DEFAULT_TEST_FILE_PATH_20_PERCENT = "data/nslkdd/KDDTest-21.txt"

def load_nslkdd(filepath: str | None=None, random_20_percent: bool=True, partition: str='train') -> tuple[pd.DataFrame, pd.Series]:
    if filepath is None:
        if partition == 'train':
            filepath = DEFAULT_TRAIN_FILE_PATH_20_PERCENT if random_20_percent else DEFAULT_TRAIN_FILE_PATH
        else:
            filepath = DEFAULT_TEST_FILE_PATH_20_PERCENT if random_20_percent else DEFAULT_TEST_FILE_PATH

    df = pd.read_csv(filepath, names=COLUMN_NAMES)
    X = df.drop(columns=['class', 'label'])
    y = df['class']

    return X, y