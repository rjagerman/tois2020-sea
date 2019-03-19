import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax


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
        
        def update(self, x, a, r):
            ips = 1.0 / max(self.cap, self.probability(x, a))
            s = np.dot(self.w[a, :], x)
            g = grad_softmax(s / self.tau)
            self.w[a, :] -= self.lr * x * g * (1.0 - 2.0 * r) * ips
        
        def draw(self, x):
            return self.baseline.draw(x)
        
        def max(self, x):
            s = np.dot(self.w, x)
            return argmax(s)
        
        def probability(self, x, a):
            return self.baseline.probability(x, a)

    return IPSPolicy


def ips_policy(k, d, baseline, lr=0.01, cap=0.05, w=None):
    w = init_weights(k, d, w)
    bl_type = numba.typeof(baseline)
    if bl_type not in _IPS_POLICY_TYPE_CACHE:
        _IPS_POLICY_TYPE_CACHE[bl_type] = _IPSPolicy(bl_type)
    return _IPS_POLICY_TYPE_CACHE[bl_type](k, d, lr, cap, baseline, w)
