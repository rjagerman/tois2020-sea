import numpy as np
import numba
from experiments.ranking.util import argsort
from rulpy.math import grad_hinge, hinge, grad_additive_dcg


_IPS_POLICY_TYPE_CACHE = {}

def _IPSPolicy(bl_type):
    @numba.jitclass([
        ('d', numba.int32),
        ('lr', numba.float64),
        ('baseline', bl_type),
        ('eta', numba.float64),
        ('cap', numba.float64),
        ('w', numba.float64[:])
    ])
    class __IPSPolicy:
        def __init__(self, d, lr, baseline, eta, cap, w):
            self.d = d
            self.lr = lr
            self.baseline = baseline
            self.eta = eta
            self.cap = cap
            self.w = w

        def update(self, dataset, index, r, c):
            x, _, _ = dataset.get(index)
            s = np.dot(x, self.w)
            for i in c:
                grad = np.zeros(self.w.shape)
                h = 1.0
                for j in range(x.shape[0]):
                    f_i = x[r[i], :]
                    f_j = x[r[j], :]
                    s_ij = s[r[i]] - s[r[j]]
                    h += hinge(s_ij)
                    g = grad_hinge(s_ij)
                    grad += (f_i - f_j) * g
                propensity = max(self.cap, (1.0 / (i + 1.0)) ** self.eta)
                self.w -= self.lr * grad * grad_additive_dcg(h) * (1.0 / propensity)

        def draw(self, x):
            return self.baseline.draw(x)

        def max(self, x):
            s = np.dot(x, self.w)
            return argsort(-s)
        
    return __IPSPolicy


def __getstate(self):
    return {
        'd': self.d,
        'lr': self.lr,
        'baseline': self.baseline,
        'eta': self.eta,
        'cap': self.cap,
        'w': self.w
    }


def __setstate(self, state):
    self.d = state['d']
    self.lr = state['lr']
    self.baseline = state['baseline']
    self.eta = state['eta']
    self.cap = state['cap']
    self.w = state['w']


def __reduce(self):
    return (IPSPolicy, (self.d, self.lr, self.baseline, self.eta, self.cap, self.w))


def __deepcopy(self):
    return IPSPolicy(self.d, self.lr, self.baseline.__deepcopy__(), self.eta, self.cap, np.copy(self.w))


def IPSPolicy(d, lr, baseline, eta=1.0, cap=0.01, w=None):
    w = np.zeros(d) if w is None else w
    bl_type = numba.typeof(baseline)
    if bl_type not in _IPS_POLICY_TYPE_CACHE:
        _IPS_POLICY_TYPE_CACHE[bl_type] = _IPSPolicy(bl_type)
    out = _IPS_POLICY_TYPE_CACHE[bl_type](d, lr, baseline, eta, cap, w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
