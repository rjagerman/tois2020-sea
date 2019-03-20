import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from rulpy.math import log_softmax
from experiments.sparse import to_dense


TYPE_UCB = 0
TYPE_THOMPSON = 1


@numba.jitclass([
    ('k', numba.int32),
    ('d', numba.int32),
    ('l2', numba.float64),
    ('alpha', numba.float64),
    ('tau', numba.float64),
    ('w', numba.float64[:,:]),
    ('b', numba.float64[:,:]),
    ('A', numba.float64[:,:,:]),
    ('A_inv', numba.float64[:,:,:]),
    ('cho', numba.float64[:,:,:]),
    ('recompute', numba.boolean[:]),
    ('draw_type', numba.int32)
])
class StatisticalPolicy:
    def __init__(self, k, d, l2, alpha, w, b, A, A_inv, cho, recompute, draw_type):
        self.k = k
        self.d = d
        self.l2 = l2
        self.alpha = alpha
        self.w = w
        self.b = b
        self.A = A
        self.A_inv = A_inv
        self.cho = cho
        self.recompute = recompute
        self.draw_type = draw_type
    
    def update(self, x, a, r):
        xd = to_dense(x)[0, :]
        x2 = xd.reshape((xd.shape[0], 1))
        self.A[a, :, :] = self.A[a, :, :] + (x2 @ x2.T)
        self.b[a, :] = self.b[a, :] + (xd * r)
        num = ((self.A_inv[a, :, :] @ x2) @ x2.T) @ self.A_inv[a, :, :]
        den = ((x2.T @ self.A_inv[a, :, :]) @ x2) + 1.0
        self.A_inv[a, :, :] -= num / den
        self.w[a, :] = np.dot(self.A_inv[a, :, :], self.b[a, :])
        self.recompute[a] = True
    
    def draw(self, x):
        xd = to_dense(x)[0, :]
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
        return argmax(np.dot(self.w, x) + self.alpha * self._bound(x))

    def _draw_thompson(self, x):
        self._update_cholesky()
        u = np.random.standard_normal(self.w.shape)
        s = np.zeros(self.k)
        for i in range(self.k):
            s[i] = np.dot(self.w[i,:] + (self.cho[i,:,:] @ u[i,:]), x)
        return argmax(s)

    def max(self, x):
        xd = to_dense(x)[0, :]
        return argmax(np.dot(self.w, xd))
    
    def probability(self, x, a):
        xd = to_dense(x)[0, :]
        if self.draw_type == TYPE_UCB:
            return self._probability_ucb(xd, a)
        elif self.draw_type == TYPE_THOMPSON:
            return self._probability_thompson(xd, a)
        else:
            raise ValueError("Unknown draw type")
    
    def _probability_ucb(self, x, a):
        if self._draw_ucb(x) == a:
            return 1.0
        else:
            return 0.0

    def _probability_thompson(self, x, a):
        self._update_cholesky()
        np.dot(x, self.w) + self._bound(x)


def statistical_policy(k, d, l2=1.0, alpha=1.0, w=None, b=None, A=None, A_inv=None,
                       cho=None, recompute=None, draw_type=TYPE_UCB, **kw_args):
    w = np.zeros((k, d), dtype=np.float64) if w is None else w
    b = np.zeros((k, d), dtype=np.float64) if b is None else b
    A = np.stack([np.identity(d, dtype=np.float64) * l2 for _ in range(k)]) if A is None else A
    A_inv = np.stack([np.identity(d, dtype=np.float64) / l2 for _ in range(k)]) if A_inv is None else A_inv
    cho = np.stack([np.zeros((d,d), dtype=np.float64) for _ in range(k)]) if cho is None else cho
    recompute = np.zeros(k, dtype=np.bool) if recompute is None else recompute
    return StatisticalPolicy(k, d, l2, alpha, w, b, A, A_inv, cho, recompute, draw_type)
