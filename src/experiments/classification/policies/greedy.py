import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('l2', numba.float64),
    ('w', numba.float64[:,:])
])
class _GreedyPolicy:
    def __init__(self, k, d, lr, l2, w):
        self.k = k
        self.d = d
        self.lr = lr
        self.l2 = l2
        self.w = w

    def update(self, dataset, index, a, r):
        x, _ = dataset.get(index)
        s = x.dot(self.w[:, a])
        loss = s - r # square loss reward compared to score (predicted reward)
        for i in range(x.nnz):
            col = x.indices[i]
            val = x.data[i]
            self.w[col, a] -= self.lr * (val * loss + self.l2 * self.w[col, a])

    def draw(self, x):
        return self.max(x)

    def max(self, x):
        return argmax(x.dot(self.w))

    def probability(self, x, a):
        s = x.dot(self.w)
        m = np.max(s)
        p = 1.0 * (s == m)
        p /= np.sum(p)
        return p[a]


def __getstate(self):
    return {
        'k': self.k,
        'd': self.d,
        'lr': self.lr,
        'w': self.w
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.lr = state['lr']
    self.w = state['w']


def __reduce(self):
    return (GreedyPolicy, (self.k, self.d), self.__getstate__())


def __deepcopy(self):
    return GreedyPolicy(self.k, self.d, self.lr, np.copy(self.w))


def GreedyPolicy(k, d, lr=0.01, w=None, **kw_args):
    w = init_weights(k, d, w)
    out = _GreedyPolicy(k, d, lr, w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
