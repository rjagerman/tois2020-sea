import numba
import numpy as np
from experiments.classification.policies.util import argmax, Policy, PolicyModel, np_copy_fn, get_np_create_fn, get_np_square_loss_update_fn, _policy_from_config
from experiments.classification.policies.greedy import action_fn as greedy_action_fn
from experiments.classification.policies.uniform import create as uniform_create
from experiments.classification.policies.boltzmann import create as create_boltzmann
from rulpy.math import log_softmax, softmax, grad_softmax


def create(config, init=None, baseline=None):
    d, k, lr, tau, cap = config.d, config.k, config.lr, config.tau, config.cap
    policy_w = create_boltzmann(config, init)
    if baseline is None:
        policy_b = uniform_create(config)
    else:
        policy_b = _policy_from_config(baseline.config, init=baseline.w)

    @numba.njit(nogil=True)
    def _create_fn():
        return (policy_w.create(), policy_b.create())

    @numba.njit(nogil=True)
    def _copy_fn(model):
        return (policy_w.copy(model[0]), policy_b.copy(model[1]))

    @numba.njit(nogil=True)
    def _update_fn(model, x, a, r):
        ips = 1.0 / max(cap, _probability_fn(model, x, a))
        s = np.dot(model[0].w, x)
        g = grad_softmax(s / tau)
        model[0].w[a, :] -= lr * x * g[a] * ips * (1.0 - 2.0 * r)

    @numba.njit(nogil=True)
    def _action_fn(model, x):
        return policy_b.draw(model[1], x)

    @numba.njit(nogil=True)
    def _probability_fn(model, x, a):
        return policy_b.probability(model[1], x, a)

    @numba.njit(nogil=True)
    def _greedy_fn(model, x):
        return policy_w.best(model[0], x)

    return Policy(
        _create_fn,
        _copy_fn,
        _update_fn,
        _action_fn,
        _greedy_fn,
        _probability_fn,
    )
