import numpy as np
import numba


@numba.njit(nogil=True)
def _numba_rng_seed(seed):
    np.random.seed(seed)


def rng_seed(seed=None):
    np.random.seed(seed)
    _numba_rng_seed(seed)


