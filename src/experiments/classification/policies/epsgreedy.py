import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights
from experiments.classification.policies.greedy import GreedyPolicy
from experiments.classification.policies.uniform import UniformPolicy
from experiments.sparse import dot_sd_mat, dot_sd_vec


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('eps', numba.float64),
    ('w', numba.float64[:,:])
])
class EpsgreedyPolicy:
    def __init__(self, k, d, lr, eps, w):
        self.k = k
        self.d = d
        self.lr = lr
        self.eps = eps
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
        if np.random.random() < self.eps:
            return np.random.randint(self.k)
        else:
            return self.max(x)
    
    def max(self, x):
        s = dot_sd_mat(x, self.w)[0, :]
        return argmax(s)
    
    def probability(self, x, a):
        s = dot_sd_mat(x, self.w)[0, :]
        up = 1.0 / float(self.k)
        m = np.max(s)
        gp = 1.0 * (s == m)
        gp /= np.sum(gp)
        gp = gp[a]
        return (self.eps) * up + (1 - self.eps) * gp


def epsgreedy_policy(k, d, lr=0.01, eps=0.05, w=None, **kw_args):
    w = init_weights(k, d, w)
    return EpsgreedyPolicy(k, d, lr, eps, w)
