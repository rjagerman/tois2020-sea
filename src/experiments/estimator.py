import numpy as np
import numba


@numba.njit(nogil=True)
def estimate_performance(dataset, ids, ys, ps, rs):
    for id in ids:
        pass

