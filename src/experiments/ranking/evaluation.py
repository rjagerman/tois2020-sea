import numpy as np
import numba
from ltrpy.evaluation.ndcg import ndcg


@numba.njit(nogil=True)
def evaluate(test, policy):
    scores = np.zeros(test.size)
    for i in range(test.size):
        x, y, q = test.get(i)
        ranking = policy.draw(x)
        scores[i] = ndcg(ranking, y)[:10][-1]
    return np.mean(scores)
