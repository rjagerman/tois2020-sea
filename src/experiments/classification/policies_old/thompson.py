import numba
import numpy as np
from math import erfc, sqrt
from experiments.classification.policies.util import argmax, Policy, PolicyModel, get_regr_create_fn, regr_copy_fn, regr_update_fn, regr_best_fn
from rulpy.linalg import regression


@numba.njit(nogil=True)
def _score_thompson(model, x):
    s = np.zeros(len(model.w))
    for a in range(len(model.w)):
        s[a] = regression.thompson(model.w[a], x)
    return s


@numba.njit(nogil=True)
def action_fn(model, x):
    return argmax(_score_thompson(model, x))


@numba.njit(nogil=True)
def probability_fn(model, x, a):
    out = 1.0
    ua, sa = regression.thompson_distribution(model.w[a], x)
    for i in range(len(model.w)):
        if i != a:
            ui, si = regression.thompson_distribution(model.w[i], x)
            out *= 1.0 - 0.5 * erfc((ua - ui) / sqrt(2*(sa + si)))
    return out


def create(config):
    return Policy(
        get_regr_create_fn(config),
        regr_copy_fn,
        regr_update_fn,
        action_fn,
        regr_best_fn,
        probability_fn,
    )
