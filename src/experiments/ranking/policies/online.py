import numpy as np
import numba
from experiments.ranking.util import argsort
from rulpy.math import grad_hinge


@numba.jitclass([
    ('d', numba.int32),
    ('lr', numba.float64),
    ('w', numba.float64[:])
])
class _OnlinePolicy:
    def __init__(self, d, lr, w):
        self.d = d
        self.lr = lr
        self.w = w

    def update(self, dataset, index, r, c):
        x, _, _ = dataset.get(index)
        s = np.dot(x, self.w)
        for i in c:
            grad = np.zeros(self.w.shape)
            for j in range(x.shape[0]):
                f_i = x[r[i], :]
                f_j = x[r[j], :]
                s_ij = s[r[i]] - s[r[j]]
                g = grad_hinge(s_ij)
                grad += (f_i - f_j) * g
            self.w -= self.lr * grad

    def draw(self, x):
        s = np.dot(x, self.w)
        return argsort(-s)


def __getstate(self):
    return {
        'd': self.d,
        'lr': self.lr,
        'w': self.w
    }

def __setstate(self, state):
    self.d = state['d']
    self.lr = state['lr']
    self.w = state['w']


def __reduce(self):
    return (Online, (self.d, self.lr, self.w))


def __deepcopy(self):
    return Online(self.d, self.lr, np.copy(self.w))


def OnlinePolicy(d, lr, w=None):
    w = np.zeros(d) if w is None else w
    out = _OnlinePolicy(d, lr, w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
