import numba
import numpy as np
from experiments.classification.policies.util import Policy, PolicyModel, np_copy_fn, get_np_create_fn, get_np_square_loss_update_fn
from experiments.classification.policies.greedy import action_fn as greedy_action


@numba.njit(nogil=True)
def _score_zero(model, x):
    return np.zeros(model.w.shape[0])


@numba.njit(nogil=True)
def _uniform(scores):
    return np.random.randint(scores.shape[0])


@numba.njit(nogil=True)
def action_fn(model, x):
    return _uniform(_score_zero(model, x))


@numba.njit(nogil=True)
def probability_fn(model, x, a):
    return 1.0 / float(model.w.shape[0])


def create(config, init=None):
    d, k, lr = config.d, config.k, config.lr

    @numba.njit(nogil=True)
    def _create_fn():
        return PolicyModel(np.zeros((k, d)), config)

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
        greedy_action,
        probability_fn,
    )
