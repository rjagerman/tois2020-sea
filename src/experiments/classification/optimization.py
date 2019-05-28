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
def optimize(train, train_indices, test, test_indices, policy):
    train_regret = 0.0
    test_regret = 0.0
    for i in range(len(train_indices)):
        x_test, y_test = test.get(test_indices[i])
        a_test = policy.draw(x_test)
        r_test = reward(x_test, y_test, a_test)
        test_regret += (1.0 - r_test)
        x, y = train.get(train_indices[i])
        a = policy.draw(x)
        r = reward(x, y, a)
        train_regret += (1.0 - r)
        policy.update(train, train_indices[i], a, r)

    return train_regret, test_regret
