import numba
import numpy as np
from experiments.classification.policies.greedy import action_fn as greedy_action, probability_fn as greedy_probability
from experiments.classification.policies.uniform import action_fn as uniform_action, probability_fn as uniform_probability
from experiments.classification.policies.util import Policy, PolicyModel, get_np_create_fn, get_np_square_loss_update_fn, np_copy_fn


def create(config, init=None):
    eps = config.eps

    @numba.njit(nogil=True)
    def _action_fn(model, x):
        if np.random.random() < eps:
            return uniform_action(model, x)
        else:
            return greedy_action(model, x)

    @numba.njit(nogil=True)
    def _probability_fn(model, x, a):
        up = uniform_probability(model, x, a)
        gp = greedy_probability(model, x, a)
        return eps * up + (1.0 - eps) * gp

    return Policy(
        get_np_create_fn(config, init),
        np_copy_fn,
        get_np_square_loss_update_fn(config),
        _action_fn,
        greedy_action,
        _probability_fn,
    )
