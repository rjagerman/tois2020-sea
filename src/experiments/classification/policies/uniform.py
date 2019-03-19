import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('w', numba.float64[:,:])
])
class UniformPolicy:
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
        return np.random.randint(self.k)
    
    def max(self, x):
        s = np.dot(self.w, x)
        return argmax(s)
    
    def probability(self, x, a):
        return 1.0 / float(self.k)


def uniform_policy(k, d, lr=0.01, w=None, **kw_args):
    w = init_weights(k, d, w)
    return UniformPolicy(k, d, lr, w)
