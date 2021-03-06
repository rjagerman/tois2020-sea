import numpy as np
import numba
from experiments.ranking.util import argsort
from rulpy.math import grad_hinge, hinge, grad_additive_dcg, additive_dcg
from rulpy.array import GrowingArray, GrowingArrayList
from llvmlite import binding
from experiments.util import ch_bound, mpeb_bound


# The jitclass below exceeds the max-name-size of llvm, because of the way
# numba names the jitclass methods. Therefore we extend the max-name-size of
# llvm here.
binding.set_option("tmp", "-non-global-value-max-name-size=4096")


_COMP_POLICY_TYPE_CACHE = {}

def _CompPolicy(bl_type):
    @numba.jitclass([
        ('d', numba.int32),
        ('lr', numba.float64),
        ('baseline', bl_type),
        ('eta', numba.float64),
        ('cap', numba.float64),
        ('w', numba.float64[:]),
        ('ips_w', numba.float64[:]),
        ('ips_w2', numba.float64[:]),
        ('ips_n', numba.int32),
        # ('history', numba.types.Tuple([
        #    numba.typeof(GrowingArray(dtype=numba.int64)),     # query ids
        #    numba.typeof(GrowingArrayList(dtype=numba.int32)), # rankings
        #    numba.typeof(GrowingArrayList(dtype=numba.int32))  # clicks
        # ])),
        ('ucb_baseline', numba.float64),
        ('lcb_w', numba.float64),
        ('confidence', numba.float64),
        ('recompute_bounds', numba.int32)
    ])
    class __CompPolicy:
        def __init__(self, d, lr, baseline, eta, cap, w, ips_w, ips_w2, ips_n, ucb_baseline, lcb_w, confidence, recompute_bounds):
            self.d = d
            self.lr = lr
            self.baseline = baseline
            self.eta = eta
            self.cap = cap
            self.w = w
            # self.history = history
            self.ips_w = ips_w
            self.ips_w2 = ips_w2
            self.ips_n = ips_n
            self.ucb_baseline = ucb_baseline
            self.lcb_w = lcb_w
            self.confidence = confidence
            self.recompute_bounds = recompute_bounds

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
            self._record_history(dataset, index, r, c)
            self.recompute_bounds += 1
            if self.recompute_bounds > 1000:
                self._recompute_bounds(dataset)
                self._update_baseline()
                self.recompute_bounds = 0

        def _update_baseline(self):
            if self.lcb_w > self.ucb_baseline:
                self.baseline.w = np.copy(self.w)

        def _record_history(self, dataset, index, ranking, clicks):
            # self.history[0].append(index)
            # self.history[1].append(ranking)
            # self.history[2].append(clicks)
            for c in clicks:
                # print("=========")
                # print(index)
                # print(dataset._starts[index])
                ips_index = dataset._starts[index] + ranking[c]
                propensity = max(self.cap, (1.0 / (c + 1)) ** self.eta)
                self.ips_w[ips_index] += (1.0 / propensity)
                self.ips_w2[ips_index] += (1.0 / propensity) ** 2
                self.ips_n += 1

        def _recompute_bounds(self, dataset):
            new_sum_mean = 0.0
            new_sum_var = 0.0
            baseline_sum_mean = 0.0
            baseline_sum_var = 0.0
            new_max = 0.0
            baseline_max = 0.0
            for qid in range(dataset.size):
                index = dataset._starts[qid]
                x, _y, _q = dataset.get(qid)
                new_r = self.max(x)
                baseline_r = self.draw(x)

                new_sorted = -additive_dcg(1.0 + np.argsort(new_r))
                baseline_sorted = -additive_dcg(1.0 + np.argsort(baseline_r))

                for i in range(x.shape[0]):
                    if self.ips_w[index] != 0.0:
                        new_sum_mean += new_sorted[i] * self.ips_w[index]
                        new_sum_var += (new_sorted[i]**2) * self.ips_w2[index]
                        baseline_sum_mean += baseline_sorted[i] * self.ips_w[index]
                        baseline_sum_var += (baseline_sorted[i]**2) * self.ips_w2[index]
                        new_max = max(new_max, new_sorted[i] * self.ips_w[index])
                        baseline_max = max(baseline_max, baseline_sorted[i] * self.ips_w[index])
                    index += 1

            new_mean = new_sum_mean / self.ips_n
            baseline_mean = baseline_sum_mean / self.ips_n
            new_var = (new_sum_var / self.ips_n) - (new_mean ** 2)
            baseline_var = (baseline_sum_var / self.ips_n) - (baseline_mean ** 2)

            self.ucb_baseline = baseline_mean + mpeb_bound(self.ips_n, self.confidence, baseline_var, baseline_max)
            self.lcb_w = new_mean - mpeb_bound(self.ips_n, self.confidence, new_var, new_max)

            # xs_new = np.zeros(self.history[2].elements, dtype=np.float64)
            # xs_baseline = np.zeros(self.history[2].elements, dtype=np.float64)
            # count = 0
            # for i in range(self.history[0].size):
            #     qid = self.history[0].get(i)
            #     ranking = self.history[1].get(i)
            #     clicks = self.history[2].get(i)
            #     x, _y, _q = dataset.get(qid)

            #     # Compute new ranking and baseline ranking
            #     s = np.dot(x, self.w)
            #     new_r = argsort(-s)
            #     baseline_r = self.baseline.draw(x)

            #     new_s = np.argsort(new_r)
            #     baseline_s = np.argsort(baseline_r)

            #     # Compute IPS-weighted avg rank
            #     for c in clicks:
            #         xc = ranking[c]
            #         propensity = max(self.cap, (1.0 / (c + 1.0)) ** self.eta)

            #         # We use negatives because we want a "reward" (higher = better)
            #         # and these are avg ranks (lower = better)
            #         xs_new[count] = -new_s[xc] / propensity
            #         xs_baseline[count] = -baseline_s[xc] / propensity
            #         count += 1

            # self.lcb_w = np.mean(xs_new) - mpeb_bound(self.confidence, xs_new)
            # self.ucb_baseline = np.mean(xs_baseline) + mpeb_bound(self.confidence, xs_baseline)
            pass

        def draw(self, x):
            return self.baseline.draw(x)

        def max(self, x):
            s = np.dot(x, self.w)
            return argsort(-s)

    return __CompPolicy


def __getstate(self):
    return {
        'd': self.d,
        'lr': self.lr,
        'baseline': self.baseline,
        'eta': self.eta,
        'cap': self.cap,
        'w': self.w,
        # 'history': (
        #     self.history[0],
        #     self.history[1],
        #     self.history[2]
        # ),
        'ips_w': self.ips_w,
        'ips_w2': self.ips_w2,
        'ips_n': self.ips_n,
        'ucb_baseline': self.ucb_baseline,
        'lcb_w': self.lcb_w,
        'confidence': self.confidence,
        'recompute_bounds': self.recompute_bounds
    }


def __setstate(self, state):
    self.d = state['d']
    self.lr = state['lr']
    self.baseline = state['baseline']
    self.eta = state['eta']
    self.cap = state['cap']
    self.w = state['w']
    # self.history = state['history']
    self.ips_w = state['ips_w']
    self.ips_w2 = state['ips_w2']
    self.ips_n = state['ips_n']
    self.ucb_baseline = state['ucb_baseline']
    self.lcb_w = state['lcb_w']
    self.confidence = state['confidence']
    self.recompute_bounds = state['recompute_bounds']


def __reduce(self):
    return (CompPolicy, (self.d, self.ips_w.shape[0], self.lr, self.baseline, self.eta, self.cap, self.w, self.ips_w, self.ips_w2, self.ips_n, self.ucb_baseline, self.lcb_w, self.confidence, self.recompute_bounds))


def __deepcopy(self):
    return CompPolicy(self.d, self.lr, self.baseline.__deepcopy__(), self.eta, self.cap, np.copy(self.w), #(
        #     self.history[0].__deepcopy__(), self.history[1].__deepcopy__(), self.history[2].__deepcopy__()
        # ),
        np.copy(self.ips_w), np.copy(self.ips_w2), self.ips_n, self.ucb_baseline, self.lcb_w, self.confidence, self.recompute_bounds)


def CompPolicy(d, pairs, lr, baseline, eta=1.0, cap=0.01, w=None, ips_w=None, ips_w2=None, ips_n=0, ucb_baseline=0.0, lcb_w=0.0, confidence=0.95, recompute_bounds=0):
    w = np.zeros(d) if w is None else w
    #if history is None:
    #    history = (GrowingArray(dtype=numba.int64), GrowingArrayList(dtype=numba.int32), GrowingArrayList(dtype=numba.int32))
    ips_w = np.zeros(pairs) if ips_w is None else ips_w
    ips_w2 = np.zeros(pairs) if ips_w2 is None else ips_w2
    bl_type = numba.typeof(baseline)
    if bl_type not in _COMP_POLICY_TYPE_CACHE:
        _COMP_POLICY_TYPE_CACHE[bl_type] = _CompPolicy(bl_type)
    out = _COMP_POLICY_TYPE_CACHE[bl_type](d, lr, baseline, eta, cap, w, ips_w, ips_w2, ips_n, ucb_baseline, lcb_w, confidence, recompute_bounds)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
