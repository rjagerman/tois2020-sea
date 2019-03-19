import numba
import numpy as np


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


def init_weights(k, d, w):
    if w is None:
        w = np.zeros((k, d), dtype=np.float64)
    if w.shape != (k, d):
        raise ValueError(f"Policy weights have incorrect shape {w.shape} != {(k, d)}")
    w = np.copy(w)
    return w
