import numba
import numpy as np
from math import erfc, sqrt
from experiments.classification.policies.util import argmax, Policy, PolicyModel, get_regr_create_fn, regr_copy_fn, regr_update_fn, regr_best_fn
from rulpy.linalg import regression


@numba.njit(nogil=True)
def _score_ucb(model, x):
    s = np.zeros(len(model.w))
    for a in range(len(model.w)):
        s[a] = regression.ucb(model.w[a], x)
    return s


@numba.njit(nogil=True)
def action_fn(model, x):
    return argmax(_score_ucb(model, x))


@numba.njit(nogil=True)
def probability_fn(model, x, a):
    if action_fn(model, x) == a:
        return 1.0
    else:
        return 0.0


def create(config):
    return Policy(
        get_regr_create_fn(config),
        regr_copy_fn,
        regr_update_fn,
        action_fn,
        regr_best_fn,
        probability_fn,
    )
