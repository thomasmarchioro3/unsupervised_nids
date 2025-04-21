import numpy as np

def _positive_sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def _negative_sigmoid(x: np.ndarray) -> np.ndarray:
    exp = np.exp(x)
    return exp / (exp + 1)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """
    Stable sigmoid implementation.
    Source: https://stackoverflow.com/questions/51976461/optimal-way-of-defining-a-numerically-stable-sigmoid-function-for-a-list-in-pyth
    
    Args:
        x (np.ndarray): Array of values.
    """

    positive = x >= 0

    # Boolean array inversion is faster than another comparison
    negative = ~positive

    # empty contains junk hence will be faster to allocate
    # Zeros has to zero-out the array after allocation, no need for that
    # See comment to the answer when it comes to dtype
    result = np.empty_like(x, dtype=float)
    result[positive] = _positive_sigmoid(x[positive])
    result[negative] = _negative_sigmoid(x[negative])

    return result

def rmse(x: np.ndarray, y: np.ndarray) -> float:
    return np.sqrt(np.mean((x-y)**2))