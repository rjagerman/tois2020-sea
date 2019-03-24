import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from experiments.sparse import from_scipy, SparseVectorList
from scipy.sparse import csr_matrix
from rulpy.array import GrowingArray
from rulpy.math import log_softmax, grad_softmax, softmax
from llvmlite import binding
binding.set_option("tmp", "-non-global-value-max-name-size=4096")


_sparse_m = numba.typeof(from_scipy(csr_matrix((0,0))))
_SEA_POLICY_TYPE_CACHE = {}

def _SEAPolicy(bl_type):
    @numba.jitclass([
        ('k', numba.int32),
        ('d', numba.int32),
        ('lr', numba.float64),
        ('cap', numba.float64),
        ('baseline', bl_type),
        ('w', numba.float64[:,:]),
        ('confidence', numba.float64),
        ('history', numba.types.Tuple([
           numba.typeof(SparseVectorList((1,))),            # vectors
           numba.typeof(GrowingArray(dtype=numba.int32)),   # actions
           numba.typeof(GrowingArray(dtype=numba.float64)), # rewards
           numba.typeof(GrowingArray(dtype=numba.float64))  # propensities
        ])),
        ('ucb_baseline', numba.float64),
        ('lcb_w', numba.float64),
        ('recompute_bounds', numba.int32)
    ])
    class SEAPolicy:
        def __init__(self, k, d, lr, cap, baseline, w, confidence, history, ucb_baseline, lcb_w):
            self.k = k
            self.d = d
            self.lr = lr
            self.cap = cap
            self.baseline = baseline
            self.w = w
            self.confidence = confidence
            self.history = history
            self.ucb_baseline = ucb_baseline
            self.lcb_w = lcb_w
            self.recompute_bounds = 0
        
        def update(self, x, a, r):
            p = max(self.cap, self.probability(x, a))
            s = x.dot(self.w)
            g = grad_softmax(s)
            for i in range(x.nnz):
                col = x.indices[i]
                val = x.data[i]
                loss = 0.5 - r # advantage loss
                self.w[col, a] -= self.lr * val * g[a] * loss / p
            self._record_history(x, a, r, p)
            self.recompute_bounds += 1
            if self.recompute_bounds >= 10000:
                self._recompute_bounds()
                self._update_baseline()
                self.recompute_bounds = 0
            
        def _record_history(self, x, a, r, p):
            self.history[0].append(x)
            self.history[1].append(a)
            self.history[2].append(r)
            self.history[3].append(p)
        
        def _update_baseline(self):
            if self.lcb_w > self.ucb_baseline:
                # replace baseline with a deepcopy of learned model
                # e.g.  `with objmode(y='intp[:]'):`
                # 
                # to support this we should supported weighted updates
                # and make learning of the new policy as a separate policy
                self.baseline.w = np.copy(self.w)

        def _recompute_bounds(self):
            n = self.history[1].size

            x_new = np.zeros(n)
            x_baseline = np.zeros(n)
            for i in range(n):
                x = self.history[0].get(i)
                a = self.history[1].get(i)
                r = self.history[2].get(i)
                p = self.history[3].get(i)
                
                s = x.dot(self.w)
                sa = 1.0 * (argmax(s) == a)
                new_p = sa * (1 - 0.05) + 0.05 * (1.0 / self.k)
                #new_p = softmax(s)[a]

                est_new = new_p * r / p
                est_baseline = self.baseline.probability(x, a) * r / p
                
                x_new[i] = est_new
                x_baseline[i] = est_baseline
            
            self.ucb_baseline = np.mean(x_baseline) + self._mpeb_bound(x_baseline)
            self.lcb_w = np.mean(x_new) - self._mpeb_bound(x_new)
        
        def _mpeb_bound(self, xs):
            n = xs.shape[0]
            C = 1.0 - self.confidence
            out = (7 * np.max(xs) * np.log(2.0 / C)) / (3 * (n - 1))
            # the n^2 factor is necessary to accurately get the MPeB bound,
            # when using the numpy variance function
            out += (1.0 / n) * np.sqrt(np.log(2.0 / C) / (n - 1) * np.var(xs) * (2.0 * n**2))
            return out

        def _ch_bound(self, xs):
            n = xs.shape[0]
            C = 1.0 - self.confidence
            return np.max(xs) * np.sqrt(np.log(1.0 / C) / (2*n))

        def draw(self, x):
            return self.baseline.draw(x)
        
        def max(self, x):
            s = x.dot(self.w)
            # log_p = log_softmax(s)
            # u = np.random.uniform(0.0, 1.0, s.shape)
            # r = np.log(-np.log(u)) - log_p
            # return argmax(-r)
            return argmax(s)
        
        def probability(self, x, a):
            return self.baseline.probability(x, a)

    return SEAPolicy


def __getstate(self):
    return {
        'k': self.k,
        'd': self.d,
        'lr': self.lr,
        'cap': self.cap,
        'baseline': self.baseline,
        'w': self.w,
        'confidence': self.confidence,
        'history': (
            self.history[0],
            self.history[1],
            self.history[2],
            self.history[3]
        ),
        'ucb_baseline': self.ucb_baseline,
        'lcb_w': self.lcb_w
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.lr = state['lr']
    self.cap = state['cap']
    self.baseline = state['baseline']
    self.w = state['w']
    self.confidence = state['confidence']
    self.history = state['history']
    self.ucb_baseline = state['ucb_baseline']
    self.lcb_w = state['lcb_w']


def __reduce(self):
    return (SEAPolicy, (self.k, self.d, self.baseline), self.__getstate__())


def __deepcopy(self):
    return SEAPolicy(self.k, self.d, self.baseline.__deepcopy__(), self.lr,
                     self.cap, np.copy(self.w), self.confidence, (
                         self.history[0].__deepcopy__(),
                         self.history[1].__deepcopy__(),
                         self.history[2].__deepcopy__(),
                         self.history[3].__deepcopy__()
                     ), self.ucb_baseline, self.lcb_w)


def SEAPolicy(k, d, baseline, lr=0.01, cap=0.05, w=None, confidence=0.95, history=None, ucb_baseline=0.0, lcb_w=0.0, **kw_args):
    w = init_weights(k, d, w)
    bl_type = numba.typeof(baseline)
    if bl_type not in _SEA_POLICY_TYPE_CACHE:
        _SEA_POLICY_TYPE_CACHE[bl_type] = _SEAPolicy(bl_type)
    history = (
        SparseVectorList((d,)),
        GrowingArray(dtype=numba.int32),
        GrowingArray(dtype=numba.float64),
        GrowingArray(dtype=numba.float64)
    ) if history is None else history
    out = _SEA_POLICY_TYPE_CACHE[bl_type](k, d, lr, cap, baseline, w, confidence, history, ucb_baseline, lcb_w)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
