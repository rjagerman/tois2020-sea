import numpy as np
import numba


@numba.njit(nogil=True)
def argsort(s):
    """
    Performs an argsort on s with random tie breaks
    """
    p = np.random.permutation(s.shape[0])
    out = np.argsort(s[p])
    return p[out]
