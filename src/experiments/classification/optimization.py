import numba
import numpy as np
from experiments.classification import dataset
from experiments.classification.util import reward


@numba.njit(nogil=True)
def optimize_supervised_hinge(train, indices, model, lr, epochs):
    w = model.w
    for e in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            x, y = train.get(i)
            s = x.dot(w)
            for j in range(train.k):
                if j != y:
                    if s[j] - s[y] + 1 > 0.0:
                        for si in range(x.nnz):
                            col = x.indices[si]
                            val = x.data[si]
                            w[col, y] += lr * val
                            w[col, j] -= lr * val


@numba.njit(nogil=True)
def optimize(train, indices, policy):
    for index in indices:
        x, y = train.get(index)
        a = policy.draw(x)
        r = reward(x, y, a)
        policy.update(train, index, a, r)
