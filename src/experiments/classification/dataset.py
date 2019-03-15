import numpy as np
import numba
from sklearn.datasets import load_svmlight_file
from collections import namedtuple


ClassificationDataset = namedtuple('ClassificationDataset', [
    'xs',
    'ys',
    'n',
    'd',
    'k'
])


@numba.njit(nogil=True)
def get(dataset, index):
    return _specialized_get(index, dataset.xs, dataset.ys)


@numba.generated_jit(nopython=True, nogil=True)
def _specialized_get(index, xs, ys):
    if isinstance(index, numba.types.Integer):
        return _get_single
    elif isinstance(index, numba.types.SliceType):
        return _get_slice
    elif isinstance(index, (numba.types.List, numba.types.Array)):
        return _get_many


@numba.njit(nogil=True)
def _get_single(index, xs, ys):
    x = xs[index, :]
    y = ys[index]
    return (x, y)


@numba.njit(nogil=True)
def _get_slice(index, xs, ys):
    out = []
    for i in range(index.start, index.stop, index.step):
        out.append(_get_single(i, xs, ys))
    return out


@numba.njit(nogil=True)
def _get_many(index, xs, ys):
    out = []
    for i in index:
        out.append(_get_single(i, xs, ys))
    return out


def load(file_path, min_size=0):
    xs, ys = load_svmlight_file(file_path)
    xs = xs.todense().A
    if min_size != 0 and min_size > xs.shape[1]:
        xs = np.hstack((xs, np.zeros((xs.shape[0], min_size - xs.shape[1]))))
    ys = ys.astype(np.int32)
    ys -= np.min(ys)
    k = np.unique(ys).shape[0]
    n = xs.shape[0]
    d = xs.shape[1]
    return ClassificationDataset(xs, ys, n, d, k)
