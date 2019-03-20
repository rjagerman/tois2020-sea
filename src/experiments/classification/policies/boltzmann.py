import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import log_softmax, grad_softmax
from experiments.sparse import dot_sd_mat, dot_sd_vec


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('tau', numba.float64),
    ('w', numba.float64[:,:])
])
class BoltzmannPolicy:
    def __init__(self, k, d, lr, tau, w):
        self.k = k
        self.d = d
        self.lr = lr
        self.tau = tau
        self.w = w
    
    def update(self, x, a, r):
        s = dot_sd_vec(x, self.w[a, :])
        row = 0
        for i in range(x.nnz):
            while x.indptr[row + 1] <= i:
                row += 1
            col = x.indices[i]
            val = x.data[i]
            self.w[a, col] -= self.lr * val * (1.0 - 2.0 * r)
    
    def draw(self, x):
        s = dot_sd_mat(x, self.w)[0, :]
        log_p = log_softmax(s / self.tau)
        u = np.random.uniform(0.0, 1.0, s.shape)
        r = np.log(-np.log(u)) - log_p
        return argmax(-r)
    
    def max(self, x):
        s = dot_sd_mat(x, self.w)[0, :]
        return argmax(s)
    
    def probability(self, x, a):
        s = dot_sd_mat(x, self.w)[0, :]
        e = np.exp(s / self.tau)
        return e[a] / np.sum(e)


def boltzmann_policy(k, d, lr=0.01, tau=1.0, w=None, **kw_args):
    w = init_weights(k, d, w)
    return BoltzmannPolicy(k, d, lr, tau, w)
