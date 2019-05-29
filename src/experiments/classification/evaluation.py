import numba
import numpy as np
from experiments.classification import dataset
from experiments.classification.util import reward


@numba.njit(nogil=True)
def evaluate(test_data, policy, vali=1.0):
    cum_r_policy = 0.0
    cum_r_best = 0.0
    nr_points = int(test_data.n * vali)
    for i in range(nr_points):
        x, y = test_data.get(i)
        a_policy = policy.draw(x)
        a_best = policy.max(x)
        r_policy = reward(x, y, a_policy)
        r_best = reward(x, y, a_best)
        cum_r_policy += r_policy
        cum_r_best += r_best
    return cum_r_policy / nr_points, cum_r_best / nr_points

