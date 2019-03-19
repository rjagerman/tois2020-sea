import numba
import numpy as np
from experiments.classification.policies.util import argmax, Policy, PolicyModel, get_np_create_fn, get_np_square_loss_update_fn, np_copy_fn


@numba.njit(nogil=True)
def action_fn(model, x):
    return argmax(np.dot(model.w, x))


@numba.njit(nogil=True)
def probability_fn(model, x, a):
    if a == action_fn(model, x):
        return 1.0
    else:
        return 0.0


def create(config, init=None):
    lr = config.lr

    @numba.njit(nogil=True)
    def _update_fn(model, x, a, r):
        w = model.w
        grad = x * (np.dot(w[a, :], x) - r)
        w[a, :] -= lr * grad

    return Policy(
        get_np_create_fn(config, init),
        np_copy_fn,
        get_np_square_loss_update_fn(config),
        action_fn,
        action_fn,
        probability_fn,
    )
