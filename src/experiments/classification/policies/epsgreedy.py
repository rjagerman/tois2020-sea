import numpy as np
import numba
from experiments.classification.policies.util import argmax, init_weights
from experiments.classification.policies.greedy import GreedyPolicy
from experiments.classification.policies.uniform import UniformPolicy


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
        s = np.dot(self.w[a, :], x)
        grad = x * (s - r)
        self.w[a, :] -= self.lr * grad
    
    def draw(self, x):
        if np.random.random() < self.eps:
            return np.random.randint(self.k)
        else:
            return self.max(x)
    
    def max(self, x):
        s = np.dot(self.w, x)
        return argmax(s)
    
    def probability(self, x, a):
        up = 1.0 / float(self.k)
        s = np.dot(self.w, x)
        m = np.max(s)
        gp = 1.0 * (s == m)
        gp /= np.sum(gp)
        gp = gp[a]
        return (self.eps) * up + (1 - self.eps) * gp


def epsgreedy_policy(k, d, lr=0.01, eps=0.05, w=None):
    w = init_weights(k, d, w)
    return EpsgreedyPolicy(k, d, lr, eps, w)
