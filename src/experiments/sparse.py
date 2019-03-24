import numpy as np
import numba
from rulpy.array import GrowingArray, GrowingArrayList


@numba.jitclass([
    ('data', numba.float64[:]),
    ('indices', numba.int32[:]),
    ('indptr', numba.int32[:]),
    ('nnz', numba.int32),
    ('shape', numba.types.UniTuple(numba.int32, 2))
])
class _SparseMatrix:
    def __init__(self, data, indices, indptr, nnz, shape):
        self.data = data
        self.indices = indices
        self.indptr = indptr
        self.nnz = nnz
        self.shape = shape

    def slice_row(self, row):
        start = self.indptr[row]
        end = self.indptr[row + 1]
        indices = self.indices[start:end]
        data = self.data[start:end]
        return _SparseVector(
            data,
            indices,
            end - start,
            (self.shape[1],)
        )


def __matrix_getstate(self):
    return {
        'data': self.data,
        'indices': self.indices,
        'indptr': self.indptr,
        'nnz': self.nnz,
        'shape': self.shape
    }


def __matrix_setstate(self, state):
    self.data = state['data']
    self.indices = state['indices']
    self.indptr = state['indptr']
    self.nnz = state['nnz']
    self.shape = state['shape']


def __matrix_reduce(self):
    return (_SparseMatrix, (self.data, self.indices, self.indptr, self.nnz, self.shape))


def from_scipy(matrix, min_d=0):
    out = _SparseMatrix(matrix.data, matrix.indices, matrix.indptr, matrix.nnz, (matrix.shape[0], max(min_d, matrix.shape[1])))
    setattr(out.__class__, '__getstate__', __matrix_getstate)
    setattr(out.__class__, '__setstate__', __matrix_setstate)
    setattr(out.__class__, '__reduce__', __matrix_reduce)
    return out


@numba.jitclass([
    ('data', numba.float64[:]),
    ('indices', numba.int32[:]),
    ('nnz', numba.int32),
    ('shape', numba.types.UniTuple(numba.int32, 1))
])
class _SparseVector:
    def __init__(self, data, indices, nnz, shape):
        self.data = data
        self.indices = indices
        self.nnz = nnz
        self.shape = shape

    def dot(self, other):
        return _sparse_vector_dot(self, other)

    def to_dense(self):
        out = np.zeros(self.shape[0])
        for i in range(self.nnz):
            out[self.indices[i]] = self.data[i]
        return out


@numba.generated_jit(nogil=True, nopython=True)
def _sparse_vector_dot(sv, other):
    if isinstance(other, numba.types.Array):
        if other.ndim == 1:
            return _sparse_vector_dot_scalar
        elif other.ndim == 2:
            return _sparse_vector_dot_vector
    raise ValueError("Sparse dot product only for 1d or 2d arrays")


@numba.njit(nogil=True)
def _sparse_vector_dot_scalar(sv, other):
    if sv.shape[0] != other.shape[0]:
        raise ValueError("Incompatible dot product shapes")
    out = 0.0
    for i in range(sv.nnz):
        out += other[sv.indices[i]] * sv.data[i]
    return out


@numba.njit(nogil=True)
def _sparse_vector_dot_vector(sv, other):
    if sv.shape[0] != other.shape[0]:
        raise ValueError("Incompatible dot product shapes")
    out = np.zeros(other.shape[1])
    for i in range(sv.nnz):
        v = sv.data[i]
        d = sv.indices[i]
        for j in range(other.shape[1]):
            out[j] += other[d, j] * v
    return out



@numba.jitclass([
    ('data', numba.typeof(GrowingArrayList(dtype=numba.float64))),
    ('indices', numba.typeof(GrowingArrayList(dtype=numba.int32))),
    ('nnzs', numba.typeof(GrowingArray(dtype=numba.int32))),
    ('shape', numba.types.UniTuple(numba.int32, 1))
])
class _SparseVectorList:
    def __init__(self, data, indices, nnzs, shape):
        self.data = data
        self.indices = indices
        self.nnzs = nnzs
        self.shape = shape

    def append(self, sv):
        self.shape = sv.shape
        self.data.append(sv.data)
        self.indices.append(sv.indices)
        self.nnzs.append(sv.nnz)
    
    def get(self, index):
        return _SparseVector(
            self.data.get(index),
            self.indices.get(index),
            self.nnzs.get(index),
            self.shape
        )
    
    def clear(self):
        self.data.clear(1)
        self.indices.clear(1)
        self.nnzs.clear(1)


def __vectorlist_getstate(self):
    return {
        'data': self.data,
        'indices': self.indices,
        'nnzs': self.nnzs,
        'shape': self.shape
    }

def __vectorlist_setstate(self, state):
    self.data = state['data']
    self.indices = state['indices']
    self.nnzs = state['nnzs']
    self.shape = state['shape']


def __vectorlist_reduce(self):
    return (SparseVectorList, (16, self.shape), self.__getstate__())


def __vectorlist_deepcopy(self):
    out = SparseVectorList(self.shape)
    out.data = self.data.__deepcopy__()
    out.indices = self.indices.__deepcopy__()
    out.nnzs = self.nnzs.__deepcopy__()
    return out


def SparseVectorList(shape, capacity=16):
    data = GrowingArrayList(capacity, dtype=numba.float64)
    indices = GrowingArrayList(capacity, dtype=numba.int32)
    nnzs = GrowingArray(capacity, dtype=numba.int32)
    out = _SparseVectorList(data, indices, nnzs, shape)
    setattr(out.__class__, '__getstate__', __vectorlist_getstate)
    setattr(out.__class__, '__setstate__', __vectorlist_setstate)
    setattr(out.__class__, '__reduce__', __vectorlist_reduce)
    setattr(out.__class__, '__deepcopy__', __vectorlist_deepcopy)
    return out
