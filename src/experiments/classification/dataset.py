import logging
import numpy as np
import numba
import json
from rulpy.pipeline.task_executor import task
from sklearn.datasets import load_svmlight_file
from scipy.sparse import csr_matrix
from collections import namedtuple
from experiments.sparse import from_scipy
from experiments.util import rng_seed

with open("conf/classification/datasets.json", "rt") as f:
    datasets = json.load(f)

_readonly_i32_1d = np.array([0], dtype=np.int32)
_readonly_i32_1d.setflags(write=False)
_sparse_m = numba.typeof(from_scipy(csr_matrix((0,0))))

@numba.jitclass([
    ('xs', _sparse_m),
    ('ys', numba.typeof(_readonly_i32_1d)),
    ('n', numba.int32),
    ('d', numba.int32),
    ('k', numba.int32)
])
class _ClassificationDataset:
    def __init__(self, xs, ys, n, d, k):
        self.xs = xs
        self.ys = ys
        self.n = n
        self.d = d
        self.k = k

    def get(self, index):
        return _specialized_get(index, self.xs, self.ys)


def ClassificationDataset(xs, ys, n, d, k):
    out = _ClassificationDataset(xs, ys, n, d, k)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    return out


def __getstate(self):
    return {
        'xs': self.xs,
        'ys': self.ys,
        'n': self.n,
        'd': self.d,
        'k': self.k
    }


def __setstate(self, state):
    self.xs = state['xs']
    self.ys = state['ys']
    self.n = state['n']
    self.d = state['d']
    self.k = state['k']


def __reduce(self):
    return (ClassificationDataset, (self.xs, self.ys, self.n, self.d, self.k))


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
    x = xs.slice_row(index)
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


@task
async def load_svm_dataset(file_path):
    logging.info(f"Loading data set from {file_path}")
    return load_svmlight_file(file_path)


@task(use_cache=True)
async def load_from_path(file_path, min_d=0, sample=1.0, seed=0,  sample_inverse=False):
    xs, ys = await load_svm_dataset(file_path)
    ys = ys.astype(np.int32)
    ys -= np.min(ys)
    if sample < 1.0:
        prng = rng_seed(seed)
        indices = prng.permutation(xs.shape[0])
        if not sample_inverse:
            indices = indices[0:int(sample*xs.shape[0])]
        else:
            indices = indices[int(sample*xs.shape[0]):]
        xs = xs[indices, :]
        ys = ys[indices]
    k = np.unique(ys).shape[0]
    n = xs.shape[0]
    d = max(min_d, xs.shape[1])
    xs = from_scipy(xs, min_d=min_d) #xs.todense().A
    ys.setflags(write=False)
    out = ClassificationDataset(xs, ys, n, d, k)
    return out


@task
async def load_train(dataset, seed=0, sample=None):
    train_path = datasets[dataset]['train']['path']
    sample = datasets[dataset]['train']['sample'] if sample is None else sample
    if sample == 1.0:
        seed = 0
    return await load_from_path(train_path, sample=sample, seed=seed)


@task
async def load_vali(dataset, seed=0):
    train_path = datasets[dataset]['vali']['path']
    sample = 1.0 - datasets[dataset]['vali']['sample']
    if sample == 1.0:
        seed = 0
    return await load_from_path(train_path, sample=sample, seed=seed, sample_inverse=True)


@task
async def load_test(dataset, seed=0):
    train = await load_train(dataset)
    test_path = datasets[dataset]['test']['path']
    sample = datasets[dataset]['test']['sample']
    if sample == 1.0:
        seed = 0
    return await load_from_path(test_path, min_d=train.d, sample=sample, seed=seed)
