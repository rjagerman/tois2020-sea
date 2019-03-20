import numpy as np
import numba
from collections import namedtuple


SparseMatrix = namedtuple('SparseMatrix', [
    'data',
    'indices',
    'indptr',
    'nnz',
    'ndim',
    'shape'
])


def from_scipy(sp, min_dim=0, write=False):
    sp.data.setflags(write=write)
    sp.indices.setflags(write=write)
    sp.indptr.setflags(write=write)
    return SparseMatrix(
        sp.data,
        sp.indices,
        sp.indptr,
        sp.nnz,
        sp.ndim,
        (sp.shape[0], max(min_dim, sp.shape[1]))
    )


@numba.njit(nogil=True)
def mat_row(s1, row):
    start = s1.indptr[row]
    end = s1.indptr[row + 1]
    sliced_indices = s1.indices[start:end]
    sliced_data = s1.data[start:end]
    return SparseMatrix(
        sliced_data,
        sliced_indices,
        np.array([0, end - start], dtype=np.int32),
        end - start,
        s1.ndim,
        (s1.shape[0] * 0 + 1, s1.shape[1])
    )


@numba.njit(nogil=True)
def dot_sd_mat(s1, d1):
    out = np.zeros((s1.shape[0], d1.shape[0]), dtype=np.float64)
    for j in range(d1.shape[0]):
        row = 0
        for i in range(s1.nnz):
            while s1.indptr[row + 1] <= i:
                row += 1
            out[row, j] += d1[j, s1.indices[i]] * s1.data[i]
    return out


@numba.njit(nogil=True)
def dot_sd_vec(s1, d1):
    out = np.zeros(s1.shape[0], dtype=np.float64)
    row = 0
    for i in range(s1.nnz):
        v = d1[s1.indices[i]] * s1.data[i]
        while s1.indptr[row + 1] <= i:
            row += 1
        out[row] += v
    return out


@numba.njit(nogil=True)
def to_dense(sp):
    out = np.zeros(sp.shape, dtype=np.float64)
    row = 0
    for i in range(sp.nnz):
        while sp.indptr[row + 1] <= i:
            row += 1
        col = sp.indices[i]
        out[row, col] = sp.data[i]
    return out
