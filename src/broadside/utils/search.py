import warnings

import numpy as np
import numpy.typing as npt


def find_nearest(x: float, vals: npt.NDArray) -> float:
    return vals[np.abs(vals - x).argmin()].item()


def find_nearest_index(x: float, vals: npt.NDArray) -> int:
    return np.abs(vals - x).argmin().item()


def find_k_minimum(arr: npt.NDArray, k: int) -> npt.NDArray:
    if k > arr.size:
        warnings.warn(f"Array smaller than k={k}")
        return arr

    if k == arr.size:
        return arr

    return arr[np.argpartition(arr, k)[:k]]


def find_k_maximum(arr: npt.NDArray, k: int) -> npt.NDArray:
    if k > arr.size:
        warnings.warn(f"Array smaller than k={k}")
        return arr

    if k == arr.size:
        return arr

    return arr[np.argpartition(arr, -k)[-k:]]
