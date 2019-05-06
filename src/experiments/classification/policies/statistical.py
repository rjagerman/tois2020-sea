import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import log_softmax
import math


TYPE_UCB = 0
TYPE_THOMPSON = 1


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('l2', numba.float64),
    ('delta', numba.float64),
    ('alpha', numba.float64),
    ('t', numba.int64),
    ('w', numba.float64[:,:]),
    ('b', numba.float64[:,:]),
    ('A', numba.float64[:,:,:]),
    ('A_inv', numba.float64[:,:,:]),
    ('cho', numba.float64[:,:,:]),
    ('recompute', numba.boolean[:]),
    ('draw_type', numba.int32)
])
class _StatisticalPolicy:
    def __init__(self, k, d, l2, alpha, t, w, b, A, A_inv, cho, recompute, draw_type):
        self.k = k
        self.d = d
        self.l2 = l2
        self.delta = alpha
        self.alpha = alpha #delta #1.0 + np.sqrt(np.log(2.0 / delta) / 2.0)
        self.t = t
        self.w = w
        self.b = b
        self.A = A
        self.A_inv = A_inv
        self.cho = cho
        self.recompute = recompute
        self.draw_type = draw_type
    
    def update(self, train, index, a, r, update_w=True):
        x, _ = train.get(index)
        xd = x.to_dense()
        x2 = xd.reshape((xd.shape[0], 1))
        self.A[a, :, :] += x2 @ x2.T
        self.b[a, :] += xd * r
        if update_w:
            num = ((self.A_inv[a, :, :] @ x2) @ x2.T) @ self.A_inv[a, :, :]
            den = ((x2.T @ self.A_inv[a, :, :]) @ x2) + 1.0
            self.A_inv[a, :, :] -= num / den
            self.w[:, a] = np.dot(self.A_inv[a, :, :], self.b[a, :])
        self.recompute[a] = True
        self.t += 1

    def update_w(self, a):
        self.A_inv[a, :, :] = np.linalg.inv(self.A[a, :, :])
        self.w[:, a] = np.dot(self.A_inv[a, :, :], self.b[a, :])
    
    def draw(self, x):
        xd = x.to_dense()
        if self.draw_type == TYPE_UCB:
            return self._draw_ucb(xd)
        elif self.draw_type == TYPE_THOMPSON:
            return self._draw_thompson(xd)
        else:
            raise ValueError("Unknown draw type")
        
    def _bound(self, x):
        x2 = x.reshape((x.shape[0], 1))
        d = np.empty(self.k)
        for i in range(self.k):
            d[i] = np.sqrt(np.diag(np.dot(np.dot(x2.T, self.A_inv[i,:,:]), x2)))[0]
        return d

    def _update_cholesky(self):
        for i in range(self.k):
            if self.recompute[i]:
                self.cho[i,:,:] = np.linalg.cholesky(self.A_inv[i,:,:])
                self.recompute[i] = False
    
    def _draw_ucb(self, x):
        s = np.dot(x, self.w) + self.alpha * self._bound(x)
        s = np.minimum(1.0, s)
        return argmax(s)

    def _draw_thompson(self, x):
        # self._update_cholesky()
        # v = 0.5 * np.sqrt(9 * self.d * np.log(max(1, self.t) / self.delta))
        # u = np.random.standard_normal(self.w.shape)
        means = np.dot(x, self.w)
        stds = self.alpha * self._bound(x)
        s =  np.empty(self.k)
        for i in range(self.k):
            s[i] = np.random.normal(means[i], stds[i])
        return argmax(s)
        # p = np.random.permutation(ps.shape[0])
        # inv_p = np.argsort(p)
        # cps = np.cumsum(ps[p])
        # r = np.random.uniform(0.0, cps[-1])
        # for i in range(self.k):
        #     if cps[i] >= r:
        #         return inv_p[i]
        # return inv_p[self.k - 1]

    def max(self, x):
        return argmax(x.dot(self.w))
    
    def probability(self, x, a):
        xd = x.to_dense()
        if self.draw_type == TYPE_UCB:
            return self._probability_ucb(xd)[a]
        elif self.draw_type == TYPE_THOMPSON:
            return self._probability_thompson(xd)[a]
        else:
            raise ValueError("Unknown draw type")
    
    def _probability_ucb(self, x):
        out = np.zeros(self.k)
        out[self._draw_ucb(x)] = 1.0
        return out

    def _probability_thompson(self, x):
        #self._update_cholesky()
        stds = self._bound(x)
        means = np.dot(x, self.w)
        out = np.zeros(self.k)
        for i in range(self.k):
            for j in range(self.k):
                if i != j:
                    out[i] += np.log(0.5 * math.erfc( (means[j] - means[i]) / math.sqrt(2.0 * (stds[i] + stds[j])) ))
        return np.exp(out)


def __getstate(self):
    return {
        'k': self.k,
        'd': self.d,
        'l2': self.l2,
        'delta': self.delta,
        'alpha': self.alpha,
        't': self.t,
        'w': self.w,
        'b': self.b,
        'A': self.A,
        'A_inv': self.A_inv,
        'cho': self.cho,
        'recompute': self.recompute,
        'draw_type': self.draw_type
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.l2 = state['l2']
    self.delta = state['delta']
    self.alpha = state['alpha']
    self.t = state['t']
    self.w = state['w']
    self.b = state['b']
    self.A = state['A']
    self.A_inv = state['A_inv']
    self.cho = state['cho']
    self.recompute = state['recompute']
    self.draw_type = state['draw_type']


def __reduce(self):
    return (StatisticalPolicy, (self.k, self.d), self.__getstate__())


def __deepcopy(self):
    return StatisticalPolicy(self.k, self.d, self.l2, self.alpha, self.t,
                             np.copy(self.w), np.copy(self.b), np.copy(self.A),
                             np.copy(self.A_inv), np.copy(self.cho),
                             np.copy(self.recompute), self.draw_type)


def StatisticalPolicy(k, d, l2=1.0, alpha=1.0, t=0, w=None, b=None, A=None, A_inv=None,
                      cho=None, recompute=None, draw_type=TYPE_UCB, **kw_args):
    w = np.zeros((d, k), dtype=np.float64) if w is None else w
    b = np.zeros((k, d), dtype=np.float64) if b is None else b
    A = np.stack([np.identity(d, dtype=np.float64) * l2 for _ in range(k)]) if A is None else A
    A_inv = np.stack([np.identity(d, dtype=np.float64) / l2 for _ in range(k)]) if A_inv is None else A_inv
    cho = np.stack([np.zeros((d,d), dtype=np.float64) for _ in range(k)]) if cho is None else cho
    recompute = np.zeros(k, dtype=np.bool) if recompute is None else recompute
    out = _StatisticalPolicy(k, d, l2, alpha, t, w, b, A, A_inv, cho, recompute, draw_type)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
