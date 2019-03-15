import numba
import numpy as np
from experiments.classification.policies.util import argmax, Policy, PolicyModel, np_copy_fn, get_np_create_fn, get_np_square_loss_update_fn
from experiments.classification.policies.greedy import action_fn as greedy_action_fn
from rulpy.math import log_softmax, softmax, grad_softmax


@numba.njit(nogil=True)
def _sample_boltzmann(scores):
    # numerically stable method to quickly sample from a log-softmax
    # distribution
    log_p = log_softmax(scores)
    u = np.random.uniform(0.0, 1.0, scores.shape)
    r = np.log(-np.log(u)) - log_p
    return argmax(-r)


def create(config, init=None):
    d, k, lr, tau = config.d, config.k, config.lr, config.tau

    @numba.njit(nogil=True)
    def _update_fn(model, x, a, r, weight=1.0):
        s = np.dot(model.w, x)
        g = grad_softmax(s / tau)
        model.w[a, :] -= lr * x * g[a] * (1.0 - 2.0 * r) * weight

    @numba.njit(nogil=True)
    def _action_fn(model, x):
        s = np.dot(model.w, x)
        return _sample_boltzmann(s / tau)

    @numba.njit(nogil=True)
    def _probability_fn(model, x, a):
        s = np.dot(model.w, x)
        e = np.exp(s / tau)
        return e[a] / np.sum(e)

    return Policy(
        get_np_create_fn(config, init),
        np_copy_fn,
        _update_fn,
        _action_fn,
        greedy_action_fn,
        _probability_fn,
    )
