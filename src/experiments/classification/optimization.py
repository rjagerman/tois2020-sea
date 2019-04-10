import numba
import numpy as np
from experiments.classification import dataset
from experiments.classification.util import reward


@numba.njit(nogil=True)
def optimize_supervised_hinge(train, indices, model, lr, epochs):
    for e in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            x, y = train.get(i)
            s = x.dot(model.w)
            for j in range(train.k):
                if j != y:
                    if s[j] - s[y] + 1 > 0.0:
                        for si in range(x.nnz):
                            col = x.indices[si]
                            val = x.data[si]
                            model.w[col, y] += lr * val
                            model.w[col, j] -= lr * val


@numba.njit(nogil=True)
def optimize_supervised_ridge(train, indices, policy, epochs=1):
    for e in range(epochs):
        for i in indices:
            x, y = train.get(i)
            for j in range(train.k):
                r = 0.0
                if j == y:
                    r = 1.0
                policy.update(train, i, j, r, update_w=False)
    for i in range(train.k):
        policy.update_w(i)


@numba.njit(nogil=True)
def optimize(train, indices, policy):
    regret = 0.0
    for index in indices:
        x, y = train.get(index)
        a = policy.draw(x)
        r = reward(x, y, a)
        regret += (1.0 - r)
        policy.update(train, index, a, r)
    return regret
