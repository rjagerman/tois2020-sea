import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import log_softmax, grad_softmax


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('lr', numba.float64),
    ('tau', numba.float64),
    ('w', numba.float64[:,:])
])
class _BoltzmannPolicy:
    def __init__(self, k, d, lr, tau, w):
        self.k = k
        self.d = d
        self.lr = lr
        self.tau = tau
        self.w = w
    
    def update(self, x, a, r):
        s = x.dot(self.w[:, a])
        for i in range(x.nnz):
            col = x.indices[i]
            val = x.data[i]
            self.w[col, a] -= self.lr * val * (s - r)
    
    def draw(self, x):
        s = x.dot(self.w)
        log_p = log_softmax(s / self.tau)
        u = np.random.uniform(0.0, 1.0, s.shape)
        r = np.log(-np.log(u)) - log_p
        return argmax(-r)
    
    def max(self, x):
        s = x.dot(self.w)
        return argmax(s)
    
    def probability(self, x, a):
        s = x.dot(self.w)
        e = np.exp(s / self.tau)
        return e[a] / np.sum(e)


def __getstate(self):
    return {
        'k': self.k,
        'd': self.d,
        'lr': self.lr,
        'tau': self.tau,
        'w': self.w
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.lr = state['lr']
    self.tau = state['tau']
    self.w = state['w']


def __reduce(self):
    return (BoltzmannPolicy, (self.k, self.d), self.__getstate__())


def BoltzmannPolicy(k, d, lr=0.01, tau=1.0, w=None, **kw_args):
    w = init_weights(k, d, w)
    out = _BoltzmannPolicy(k, d, lr, tau, w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    return out
