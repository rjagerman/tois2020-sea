import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights
from experiments.sparse import dot_sd_mat, dot_sd_vec


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
        s = dot_sd_vec(x, self.w[a, :])[0]
        row = 0
        for i in range(x.nnz):
            while x.indptr[row + 1] <= i:
                row += 1
            col = x.indices[i]
            val = x.data[i]
            self.w[a, col] -= self.lr * val * (s - r)
    
    def draw(self, x):
        return self.max(x)
    
    def max(self, x):
        s = dot_sd_mat(x, self.w)[0, :]
        return argmax(s)
    
    def probability(self, x, a):
        s = dot_sd_mat(x, self.w)[0, :]
        m = np.max(s)
        p = 1.0 * (s == m)
        p /= np.sum(p)
        return p[a]


def greedy_policy(k, d, lr=0.01, w=None, **kw_args):
    w = init_weights(k, d, w)
    return GreedyPolicy(k, d, lr, w)
