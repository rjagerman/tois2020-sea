import logging
import numpy as np
import numba
import json
from rulpy.pipeline.task_executor import task
from sklearn.datasets import load_svmlight_file
from collections import namedtuple
from experiments.sparse import from_scipy, mat_row
from experiments.util import rng_seed

with open("conf/datasets.json", "rt") as f:
    datasets = json.load(f)


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
    x = mat_row(xs, index)
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
async def load(file_path, min_size=0, sample=1.0, seed=0):
    xs, ys = await load_svm_dataset(file_path)
    ys = ys.astype(np.int32)
    ys -= np.min(ys)
    if sample < 1.0:
        prng = rng_seed(seed)
        indices = prng.permutation(xs.shape[0])[0:int(sample*xs.shape[0])]
        xs = xs[indices, :]
        ys = ys[indices]
    k = np.unique(ys).shape[0]
    n = xs.shape[0]
    d = xs.shape[1]
    xs = from_scipy(xs, min_size) #xs.todense().A
    ys.setflags(write=False)
    return ClassificationDataset(xs, ys, n, d, k)


@task
async def load_train(dataset, seed=0):
    train_path = datasets[dataset]['train']['path']
    sample = datasets[dataset]['train']['sample']
    if sample == 1.0:
        seed = 0
    return await load(train_path, sample=sample, seed=seed)


@task
async def load_test(dataset, seed=0):
    train = await load_train(dataset)
    test_path = datasets[dataset]['test']['path']
    sample = datasets[dataset]['test']['sample']
    if sample == 1.0:
        seed = 0
    return await load(test_path, min_size=train.d, sample=sample, seed=seed)
