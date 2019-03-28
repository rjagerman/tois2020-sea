import numpy as np
import numba
from rulpy.math import grad_hinge


@numba.njit(nogil=True)
def optimize_supervised(train, indices, policy, lr, epochs):
    for e in range(epochs):
        for i in range(indices.shape[0]):
            x, y, q = train.get(indices[i])
            s = np.dot(x, policy.w)
            for j in range(y.shape[0]):
                for k in range(y.shape[0]):
                    # pairwise difference to learn from
                    if y[j] > y[k]:
                        grad = (x[j, :] - x[k, :]) * grad_hinge(s[j] - s[k])
                        policy.w -= lr * grad


@numba.njit(nogil=True)
def optimize(train, indices, policy, behavior):
    for i in indices:
        x, y, q = train.get(i)
        r = policy.draw(x)
        c = behavior.simulate(r, y)
        cc = np.where(c > 0)[0]
        policy.update(train, i, r, cc)
