import numpy as np
import numba
import os
import json


@numba.njit(nogil=True)
def _numba_rng_seed(seed):
    np.random.seed(seed)


def rng_seed(seed=None):
    np.random.seed(seed)
    _numba_rng_seed(seed)
    return np.random.RandomState(seed)


def get_evaluation_points(iterations, evaluations, scale):
    if scale == 'log':
        evaluations = 1 + int(np.log10(iterations)) * int(evaluations / np.log10(iterations))
        return np.unique(np.concatenate((np.zeros(1, dtype=np.int32), np.geomspace(1, iterations, evaluations, dtype=np.int32))))
    elif scale == 'lin':
        return np.concatenate((np.arange(0, iterations, iterations / evaluations, dtype=np.int32), np.array([iterations], dtype=np.int32)))
    else:
        return np.array([iterations], dtype=np.int32)


def mkdir_if_not_exists(path):
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
