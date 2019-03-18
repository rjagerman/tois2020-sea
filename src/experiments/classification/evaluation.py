import numba
import numpy as np
from experiments.classification import dataset
from experiments.classification.util import reward


@numba.njit(nogil=True)
def evaluate(test_data, model, policy):
    cum_r_policy = 0.0
    cum_r_best = 0.0
    for i in range(test_data.n):
        x, y = dataset.get(test_data, i)
        a_policy = policy.draw(model, x)
        a_best = policy.best(model, x)
        r_policy = reward(x, y, a_policy)
        r_best = reward(x, y, a_best)
        cum_r_policy += r_policy
        cum_r_best += r_best
    return cum_r_policy / test_data.n, cum_r_best / test_data.n
