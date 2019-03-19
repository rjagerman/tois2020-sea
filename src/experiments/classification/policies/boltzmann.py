import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import log_softmax


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
        s = np.dot(self.w[a, :], x)
        g = grad_softmax(s / self.tau)
        self.w[a, :] -= self.lr * x * g * (1.0 - 2.0 * r)
    
    def draw(self, x):
        s = np.dot(self.w, x)
        log_p = log_softmax(s / self.tau)
        u = np.random.uniform(0.0, 1.0, s.shape)
        r = np.log(-np.log(u)) - log_p
        return argmax(-r)
    
    def max(self, x):
        s = np.dot(self.w, x)
        return argmax(s)
    
    def probability(self, x, a):
        s = np.dot(self.w, x)
        e = np.exp(s / self.tau)
        return e[a] / np.sum(e)


def boltzmann_policy(k, d, lr=0.01, tau=1.0, w=None, **kw_args):
    w = init_weights(k, d, w)
    return BoltzmannPolicy(k, d, lr, tau, w)
