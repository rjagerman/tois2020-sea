import numpy as np
import numba
from experiments.classification.policies.util import init_weights, argmax
from experiments.sparse import from_scipy, SparseVectorList
from experiments.util import mpeb_bound
from scipy.sparse import csr_matrix
from rulpy.array import GrowingArray
from rulpy.math import log_softmax, grad_softmax, softmax
from llvmlite import binding


# The jitclass below exceeds the max-name-size of llvm, because of the way
# numba names the jitclass methods. Therefore we extend the max-name-size of
# llvm here.
binding.set_option("tmp", "-non-global-value-max-name-size=4096")


_SEA_POLICY_TYPE_CACHE = {}


def _SEAPolicy(bl_type):
    @numba.jitclass([
        ('k', numba.int32),
        ('d', numba.int32),
        ('n', numba.int32),
        ('lr', numba.float64),
        ('cap', numba.float64),
        ('baseline', bl_type),
        ('w', numba.float64[:,:]),
        ('confidence', numba.float64),
        ('ips_w', numba.float64[:,:]),
        ('ips_w2', numba.float64[:,:]),
        ('ips_n', numba.int64),
        # ('history', numba.types.Tuple([
        #    numba.typeof(GrowingArray(dtype=numba.int32)),   # vectors
        #    numba.typeof(GrowingArray(dtype=numba.int32)),   # actions
        #    numba.typeof(GrowingArray(dtype=numba.float64)), # rewards
        #    numba.typeof(GrowingArray(dtype=numba.float64))  # propensities
        # ])),
        ('ucb_baseline', numba.float64),
        ('lcb_w', numba.float64),
        ('recompute_bounds', numba.int32)
    ])
    class SEAPolicy:
        def __init__(self, k, d, n, lr, cap, baseline, w, confidence, ips_w, ips_w2, ips_n, ucb_baseline, lcb_w, recompute_bounds):
            self.k = k
            self.d = d
            self.n = n
            self.lr = lr
            self.cap = cap
            self.baseline = baseline
            self.w = w
            self.confidence = confidence
            # self.history = history
            self.ips_w = ips_w
            self.ips_w2 = ips_w2
            self.ips_n = ips_n
            self.ucb_baseline = ucb_baseline
            self.lcb_w = lcb_w
            self.recompute_bounds = recompute_bounds
        
        def update(self, dataset, index, a, r):
            x, _ = dataset.get(index)
            p = max(self.cap, self.probability(x, a))
            s = x.dot(self.w)
            g = grad_softmax(s)
            loss = 0.5 - r # advantage loss
            for i in range(x.nnz):
                col = x.indices[i]
                val = x.data[i]
                self.w[col, a] -= self.lr * val * g[a] * loss / p
            self._record_history(index, a, r, p)
            self.recompute_bounds += 1
            if self.recompute_bounds >= 1000:
                self._recompute_bounds(dataset)
                self._update_baseline()
                self.recompute_bounds = 0
            
        def _record_history(self, index, a, r, p):
            # self.history[0].append(index)
            # self.history[1].append(a)
            # self.history[2].append(r)
            # self.history[3].append(p)
            self.ips_w[index, a] += r / p
            self.ips_w2[index, a] += (r / p) ** 2
            self.ips_n += 1
        
        def _update_baseline(self):
            if self.lcb_w > self.ucb_baseline:
                # replace baseline with a deepcopy of learned model
                # e.g.  `with objmode(y='intp[:]'):`
                # 
                # to support this we should supported weighted updates
                # and make learning of the new policy as a separate policy
                self.baseline.w = np.copy(self.w)

        def _recompute_bounds(self, dataset):
            new_sum_mean = 0.0
            new_sum_var = 0.0
            baseline_sum_mean = 0.0
            baseline_sum_var = 0.0
            new_max = 0.0
            baseline_max = 0.0
            for index in range(self.n):
                for a in range(self.k):
                    if self.ips_w[index, a] != 0.0:
                        x, _ = dataset.get(index)
                        s = x.dot(self.w)
                        sa = 1.0 * (argmax(s) == a)
                        new_p = sa * (1 - 0.05) + 0.05 * (1.0 / self.k)
                        baseline_p = self.baseline.probability(x, a)
                        new_sum_mean += new_p * self.ips_w[index, a]
                        new_sum_var += new_p**2 * self.ips_w2[index, a]
                        baseline_sum_mean += baseline_p * self.ips_w[index, a]
                        baseline_sum_var += baseline_p**2 * self.ips_w2[index, a]
                        new_max = max(new_max, new_p * self.ips_w[index, a])
                        baseline_max = max(baseline_max, baseline_p * self.ips_w[index, a])
            new_mean = new_sum_mean / self.ips_n
            baseline_mean = baseline_sum_mean / self.ips_n

            new_var = (new_sum_var / self.ips_n) - (new_mean ** 2)
            baseline_var = (baseline_sum_var / self.ips_n) - (baseline_mean ** 2)

            self.ucb_baseline = baseline_mean + mpeb_bound(self.ips_n, self.confidence, baseline_var, baseline_max)
            self.lcb_w = new_mean - mpeb_bound(self.ips_n, self.confidence, new_var, new_max)
                        
            # n = self.history[1].size

            # x_new = np.zeros(n)
            # x_baseline = np.zeros(n)
            # for i in range(n):
            #     index = self.history[0].get(i)
            #     a = self.history[1].get(i)
            #     r = self.history[2].get(i)
            #     p = self.history[3].get(i)
            #     x, _ = dataset.get(index)
            
            #     s = x.dot(self.w)
            #     sa = 1.0 * (argmax(s) == a)
            #     new_p = sa * (1 - 0.05) + 0.05 * (1.0 / self.k)

            #     est_new = new_p * r / p
            #     est_baseline = self.baseline.probability(x, a) * r / p
            
            #     x_new[i] = est_new
            #     x_baseline[i] = est_baseline
            
            # self.ucb_baseline = np.mean(x_baseline) + self._mpeb_bound(x_baseline)
            # self.lcb_w = np.mean(x_new) - self._mpeb_bound(x_new)

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
        'n': self.n,
        'lr': self.lr,
        'cap': self.cap,
        'baseline': self.baseline,
        'w': self.w,
        'confidence': self.confidence,
        # 'history': (
        #     self.history[0],
        #     self.history[1],
        #     self.history[2],
        #     self.history[3]
        # ),
        'ips_w': self.ips_w,
        'ips_w2': self.ips_w2,
        'ips_n': self.ips_n,
        'ucb_baseline': self.ucb_baseline,
        'lcb_w': self.lcb_w,
        'recompute_bounds': self.recompute_bounds
    }


def __setstate(self, state):
    self.k = state['k']
    self.d = state['d']
    self.n = state['n']
    self.lr = state['lr']
    self.cap = state['cap']
    self.baseline = state['baseline']
    self.w = state['w']
    self.confidence = state['confidence']
    #self.history = state['history']
    self.ips_w = state['ips_w']
    self.ips_w2 = state['ips_w2']
    self.ips_n = state['ips_n']
    self.ucb_baseline = state['ucb_baseline']
    self.lcb_w = state['lcb_w']
    self.recompute_bounds = state['recompute_bounds']


def __reduce(self):
    return (SEAPolicy, (self.k, self.d, self.n, self.baseline), self.__getstate__())


def __deepcopy(self):
    return SEAPolicy(self.k, self.d, self.n, self.baseline.__deepcopy__(), self.lr,
                     self.cap, np.copy(self.w), self.confidence, #(
                    #      self.history[0].__deepcopy__(),
                    #      self.history[1].__deepcopy__(),
                    #      self.history[2].__deepcopy__(),
                    #      self.history[3].__deepcopy__()
                    #  ),
                    np.copy(self.ips_w), np.copy(self.ips_w2), self.ips_n, self.ucb_baseline, self.lcb_w, self.recompute_bounds)


def SEAPolicy(k, d, n, baseline, lr=0.01, cap=0.05, w=None, confidence=0.95, ips_w=None, ips_w2=None, ips_n=0, ucb_baseline=0.0, lcb_w=0.0, recompute_bounds=0, **kw_args):
    w = init_weights(k, d, w)
    bl_type = numba.typeof(baseline)
    if bl_type not in _SEA_POLICY_TYPE_CACHE:
        _SEA_POLICY_TYPE_CACHE[bl_type] = _SEAPolicy(bl_type)
    # history = (
    #     GrowingArray(dtype=numba.int32),
    #     GrowingArray(dtype=numba.int32),
    #     GrowingArray(dtype=numba.float64),
    #     GrowingArray(dtype=numba.float64)
    # ) if history is None else history
    ips_w = np.zeros((n, k)) if ips_w is None else ips_w
    ips_w2 = np.zeros((n, k)) if ips_w2 is None else ips_w2
    out = _SEA_POLICY_TYPE_CACHE[bl_type](k, d, n, lr, cap, baseline, w, confidence, ips_w, ips_w2, ips_n, ucb_baseline, lcb_w, recompute_bounds)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
