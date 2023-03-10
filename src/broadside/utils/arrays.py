import math

import numpy as np
import numpy.typing as npt


def square_concat(arrays: list[npt.NDArray]) -> npt.NDArray:
    first = arrays[0]
    h, w = first.shape[-2:]

    n = len(arrays)

    ny = int(math.ceil(math.sqrt(n) * w / h))
    nx = int(math.ceil(n / ny))

    dest_shape = (first.shape[:-2]) + (ny * h, nx * w)
    dest = np.empty(dest_shape, dtype=first.dtype)

    for iy in range(ny):
        for ix in range(nx):
            i = nx * iy + ix
            if not i < n:
                a = np.zeros_like(arrays[0])
            else:
                a = arrays[i]

            dest[..., h * iy : h * (iy + 1), w * ix : w * (ix + 1)] = a

    return dest
