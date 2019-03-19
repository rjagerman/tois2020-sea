import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('w', numba.float64[:,:])
])
class GreedyPolicy:
    def __init__(self, k, d, lr, w):
        self.k = k
        self.d = d
        self.lr = lr
        self.w = w
    
    def update(self, x, a, r):
        s = np.dot(self.w[a, :], x)
        grad = x * (s - r)
        self.w[a, :] -= self.lr * grad
    
    def draw(self, x):
        return self.max(x)
    
    def max(self, x):
        s = np.dot(self.w, x)
        return argmax(s)
    
    def probability(self, x, a):
        s = np.dot(self.w, x)
        m = np.max(s)
        p = 1.0 * (s == m)
        p /= np.sum(p)
        return p[a]


def greedy_policy(k, d, lr=0.01, w=None, **kw_args):
    w = init_weights(k, d, w)
    return GreedyPolicy(k, d, lr, w)
