import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import grad_softmax, log_softmax, grad_log_softmax


_IPS_POLICY_TYPE_CACHE = {}

def _IPSPolicy(bl_type):
    @numba.jitclass([
        ('k', numba.int32),
        ('d', numba.int32),
        ('lr', numba.float64),
        ('cap', numba.float64),
        ('baseline', bl_type),
        ('w', numba.float64[:,:])
    ])
    class IPSPolicy:
        def __init__(self, k, d, lr, cap, baseline, w):
            self.k = k
            self.d = d
            self.lr = lr
            self.cap = cap
            self.baseline = baseline
            self.w = w
        
        def update(self, dataset, index, a, r):
            x, _ = dataset.get(index)
            p = max(self.cap, self.probability(x, a))
            s = x.dot(self.w)
            g = grad_softmax(s)
            for i in range(x.nnz):
                col = x.indices[i]
                val = x.data[i]
                loss = 0.5 - r # advantage loss
                self.w[col, a] -= self.lr * val * g[a] * loss / p
            # ips = 1.0 / max(self.cap, self.probability(x, a))
            # s = x.dot(self.w)
            # g = grad_softmax(s)
            # for i in range(x.nnz):
            #     col = x.indices[i]
            #     val = x.data[i]
            #     self.w[col, a] -= self.lr * val * g[a] * (1.0 - 2.0 * r) * ips
        
        def draw(self, x):
            return self.baseline.draw(x)
        
        def max(self, x):
            s = x.dot(self.w)
            return argmax(s)
        
        def probability(self, x, a):
            return self.baseline.probability(x, a)

    return IPSPolicy


def __getstate(self):
    return {
        'k': self.k,
        'd': self.d,
        'lr': self.lr,
        'cap': self.cap,
        'baseline': self.baseline,
        'w': self.w
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.lr = state['lr']
    self.cap = state['cap']
    self.baseline = state['baseline']
    self.w = state['w']


def __reduce(self):
    return (IPSPolicy, (self.k, self.d, self.baseline), self.__getstate__())


def __deepcopy(self):
    return IPSPolicy(self.k, self.d, self.baseline.__deepcopy__(), self.lr, self.cap, np.copy(self.w))


def IPSPolicy(k, d, baseline, lr=0.01, cap=0.05, w=None, **kw_args):
    w = init_weights(k, d, w)
    bl_type = numba.typeof(baseline)
    if bl_type not in _IPS_POLICY_TYPE_CACHE:
        _IPS_POLICY_TYPE_CACHE[bl_type] = _IPSPolicy(bl_type)
    out = _IPS_POLICY_TYPE_CACHE[bl_type](k, d, lr, cap, baseline, w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
