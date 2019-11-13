import numpy as np
import numba
from experiments.ranking.util import argsort



@numba.jitclass([
    ('d', numba.int32),
    ('lr', numba.float64),
    ('delta', numba.float64),
    ('w', numba.float64[:]),
    ('next_u', numba.float64[:]),
    ('should_update', numba.types.Set(numba.int32))
])
class _DuelingBanditPolicy:
    def __init__(self, d, lr, delta, w, next_u, should_update):
        self.d = d
        self.lr = lr
        self.delta = delta
        self.w = w
        self.next_u = next_u
        self.should_update = should_update

    def update(self, dataset, index, r, c):
        # Count clicks
        clicks_for_update = 0
        for c_i in c:
            if c_i in self.should_update:
                clicks_for_update += 1
            else:
                clicks_for_update -= 1

        # Perform update if the new u is better
        if clicks_for_update > 0:
            self.w += self.lr * self.next_u

    def draw(self, x):
        # Sample new model from unit sphere and store it
        u = np.random.randn(self.w.shape[0])
        u /= np.linalg.norm(u, axis=0)
        self._next_u = u

        # Compute new weight (projected) vector
        next_w = self.w + self.delta * u
        next_w /= np.linalg.norm(next_w, axis=0)

        # Compute rankings for old model (w) and new model (next_w)
        r1 = argsort(-np.dot(x, self.w))
        r2 = argsort(-np.dot(x, next_w))

        # Generate interleaved ranking
        int_ranking, t1, t2 = tdi(r1, r2)

        # Store which ranks would result in an update to the model
        self.should_update = t2

        # Return ranking
        return int_ranking

    def max(self, x):
        s = np.dot(x, self.w)
        return argsort(-s)


@numba.njit(nogil=True)
def tdi(r1, r2):
    # Set pointers to 0 for the start of both rankings r1 and r2
    c_1 = 0
    c_2 = 0

    # Create teams and an interleaved ranking for output
    int_r = np.zeros(r1.shape[0] + r2.shape[0])
    int_size = 0
    int_set = set()
    t1 = set()
    t2 = set()

    # Main TDI algorithm
    while c_1 < len(r1) and c_2 < len(r2):
        if t1_size < t2_size or (t1_size == t2_size and np.random.rand() > 0.5):
            int_r[int_size] = r1[c_1]
            int_set.add(r1[c_1])
            t1.add(r1[c_1])
            int_size += 1
            c_1 += 1
        else:
            int_r[int_size] = r2[c_2]
            int_set.add(r2[c_2])
            t2.add(r2[c_2])
            int_size += 1
            c_2 += 1

        # Increment pointers until we reach a document not yet
        # in the final ranking
        while c_1 < len(r1) and r1[c_1] in iset:
            c_1 += 1
        while c_2 < len(r2) and r2[c_2] in iset:
            c_2 += 1

    # Return interleaved results and the two teams
    return int_r[0:int_size], t1, t2


def __getstate(self):
    return {
        'd': self.d,
        'lr': self.lr,
        'delta': self.delta,
        'w': self.w,
        'next_u': self.next_u,
        'should_update': self.should_update
    }

def __setstate(self, state):
    self.d = state['d']
    self.lr = state['lr']
    self.delta = state['delta']
    self.w = state['w']
    self.next_u = state['next_u']
    self.should_update = state['should_update']


def __reduce(self):
    return (DuelingBanditPolicy, (self.d, self.lr, self.delta, self.w,
                                  self.next_u, self.should_update))


def __deepcopy(self):
    return DuelingBanditPolicy(self.d, self.lr, self.delta, np.copy(self.w),
                               np.copy(self.next_u),
                               self.should_update.copy())


def DuelingBanditPolicy(d, lr, delta=0.1, w=None, next_u=None,
                        should_update=None):
    w = np.zeros(d) if w is None else w
    next_u = np.zeros(d) if next_u is None else next_u
    should_update = set() if should_update is None else should_update
    out = _DuelingBanditPolicy(d, lr, delta, w, next_u, should_update)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
