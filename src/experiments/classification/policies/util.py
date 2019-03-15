import numba
import numpy as np
from rulpy.linalg import regression
from collections import namedtuple


PolicyConfig = namedtuple('PolicyConfig', [
    'd',
    'k',
    'strategy',
    'lr',
    'l2',
    'eps',
    'tau',
    'alpha',
    'cap'
])


Policy = namedtuple('Policy', [
    'create',
    'copy',
    'update',
    'draw',
    'best',
    'probability'
])


PolicyModel = namedtuple('PolicyModel', [
    'w',
    'config'
])


@numba.njit(nogil=True)
def argmax(scores):
    best_score = -np.inf
    best_action = 0
    c = 2.0
    for a in range(len(scores)):
        s = scores[a]
        if s > best_score:
            best_action = a
            best_score = s
        elif s == best_score and np.random.random() < 1 / c:
            best_action = a
            best_score = s
            c += 1.0
    return best_action


def get_np_create_fn(config, init=None):
    if init is None or init.shape != (config.k, config.d):
        init = np.zeros((config.k, config.d))
    def _create_fn():
        return PolicyModel(np.copy(init), config)
    return numba.njit(nogil=True)(_create_fn)


def get_np_square_loss_update_fn(config):
    lr = config.lr
    def _update_fn(model, x, a, r):
        w = model.w
        grad = x * (np.dot(w[a, :], x) - r)
        w[a, :] -= lr * grad
    return numba.njit(nogil=True)(_update_fn)


@numba.njit(nogil=True)
def np_copy_fn(model):
    return PolicyModel(np.copy(model.w), model.config)


def get_regr_create_fn(config):
    d, k, l2 = config.d, config.k, config.l2
    def _create_fn():
        w = [regression.ridge_regression(d, l2) for _ in range(k)]
        return PolicyModel(w, config)
    return numba.njit(nogil=True)(_create_fn)


@numba.njit(nogil=True)
def regr_copy_fn(model):
    w = [regression.copy(m) for m in model.w]
    return PolicyModel(w, model.config)


@numba.njit(nogil=True)
def regr_update_fn(model, x, a, r):
    regression.update(model.w[a], x, r)


@numba.njit(nogil=True)
def regr_best_fn(model, x):
    s = np.zeros(len(model.w))
    for a in range(len(s)):
        s[a] = regression.predict(model.w[a], x)
    return argmax(s)


def _policy_from_config(config, init=None, baseline=None):
    from experiments.classification.policies.boltzmann import create as create_boltzmann
    from experiments.classification.policies.epsgreedy import create as create_epsgreedy
    from experiments.classification.policies.greedy import create as create_greedy
    from experiments.classification.policies.thompson import create as create_thompson
    from experiments.classification.policies.ucb import create as create_ucb
    from experiments.classification.policies.uniform import create as create_uniform
    from experiments.classification.policies.ips import create as create_ips
    # if config not in _POLICIES:
    #     _POLICIES[config] = {
    #         0: lambda: create_boltzmann(config),
    #         1: lambda: create_epsgreedy(config),
    #         2: lambda: create_greedy(config),
    #         3: lambda: create_thompson(config),
    #         4: lambda: create_ucb(config),
    #         5: lambda: create_uniform(config),
    #         6: lambda: create_ips(config, baseline)
    #     }[config.strategy]()
    # return _POLICIES[config]
    return {
        0: lambda: create_boltzmann(config, init),
        1: lambda: create_epsgreedy(config, init),
        2: lambda: create_greedy(config, init),
        3: lambda: create_thompson(config),
        4: lambda: create_ucb(config),
        5: lambda: create_uniform(config, init),
        6: lambda: create_ips(config, init=init, baseline=baseline)
    }[config.strategy]()
