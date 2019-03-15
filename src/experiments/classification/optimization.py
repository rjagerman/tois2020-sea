import numpy as np
import numba
from rulpy.linalg import regression
from collections import namedtuple


Optimizer = namedtuple('Optimizer',[
    'create',
    'update',
    'copy'
])


LinearModel = namedtuple('LinearModel', [
    'w'
])


def create_sgd_optimizer(d, lr=0.1):
    def _sgd_create():
        return LinearModel(np.zeros(d))
    def _sgd_update(model, grad):
        w = model.w
        w -= lr * grad
    return Optimizer(
        numba.njit(nogil=True)(_sgd_create),
        numba.njit(nogil=True)(_sgd_update),
        _sgd_copy
    )


def create_analytical_optimizer(d, l2):
    def _analytical_create():
        return regression.ridge_regression(d, l2)
    return Optimizer(
        numba.njit(nogil=True)(_analytical_create),
        _analytical_update,
        _analytical_copy
    )


def create_optimizer(optimizer, d, lr=0.1, l2=1.0):
    return {
        'sgd': create_sgd_optimizer(d, lr),
        'analytical': create_analytical_optimizer(d, l2)
    }[optimizer]


@numba.njit(nogil=True)
def _analytical_update(model, x, y):
    regression.update(model, x, y)


@numba.njit(nogil=True)
def _sgd_copy(model):
    return LinearModel(np.copy(model.w))


@numba.njit(nogil=True)
def _analytical_copy(model):
    return regression.copy(model)
