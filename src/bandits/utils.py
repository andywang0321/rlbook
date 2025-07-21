"""
utils.py
"""

import numpy as np
from numpy.typing import NDArray


def argmax(x: NDArray[np.float64]) -> NDArray[np.intp]:
    max_val: float = max(x)
    indices: list[int] = [i for i, val in enumerate(x) if val == max_val]
    return np.array(indices)


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
