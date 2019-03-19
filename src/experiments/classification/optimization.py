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
            grad = np.zeros((train.k, train.d))
            x, y = dataset.get(train, i)
            s = np.dot(w, x)
            for j in range(train.k):
                if j != y:
                    if s[j] - s[y] + 1 > 0.0:
                        grad[y, :] -= x
                        grad[j, :] += x
            w -= lr * grad


@numba.njit(nogil=True)
def optimize(train, indices, policy):
    for index in indices:
        x, y = dataset.get(train, index)
        a = policy.draw(x)
        r = reward(x, y, a)
        policy.update(x, a, r)

