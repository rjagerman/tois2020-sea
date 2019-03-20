import numba
import numpy as np
from experiments.classification import dataset
from experiments.classification.util import reward
from experiments.sparse import dot_sd_mat, to_dense


@numba.njit(nogil=True)
def optimize_supervised_hinge(train, indices, model, lr, epochs):
    w = model.w
    for e in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            x, y = dataset.get(train, i)
            s = dot_sd_mat(x, w)[0, :]
            for j in range(train.k):
                if j != y:
                    if s[j] - s[y] + 1 > 0.0:
                        for si in range(x.nnz):
                            col = x.indices[si]
                            val = x.data[si]
                            w[y, col] += lr * val
                            w[j, col] -= lr * val


@numba.njit(nogil=True)
def optimize(train, indices, policy):
    for index in indices:
        x, y = dataset.get(train, index)
        a = policy.draw(x)
        r = reward(x, y, a)
        policy.update(x, a, r)

