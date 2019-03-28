import numpy as np
import numba
from ltrpy.evaluation.ndcg import ndcg


@numba.njit(nogil=True)
def evaluate(test, policy):
    scores = np.zeros(test.size)
    scores2 = np.zeros(test.size)
    for i in range(test.size):
        x, y, q = test.get(i)
        ranking = policy.draw(x)
        ranking2 = policy.max(x)
        scores[i] = ndcg(ranking, y)[:10][-1]
        scores2[i] = ndcg(ranking2, y)[:10][-1]
    return np.mean(scores), np.mean(scores2)


@numba.njit(nogil=True)
def evaluate_fraction(test, policy, fraction):
    size = int(fraction * test.size)
    indices = np.random.permutation(test.size)[0:size]
    scores = np.zeros(size)
    scores2 = np.zeros(size)
    for i in range(indices.shape[0]):
        x, y, q = test.get(indices[i])
        ranking = policy.draw(x)
        ranking2 = policy.max(x)
        scores[i] = ndcg(ranking, y)[:10][-1]
        scores2[i] = ndcg(ranking2, y)[:10][-1]
    return np.mean(scores), np.mean(scores2)
