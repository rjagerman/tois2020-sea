import numpy as np
import numba
from experiments.ranking.util import argsort
from ltrpy.ranker import inorder



@numba.jitclass([
    ('d', numba.int32),
    ('lr', numba.float64),
    ('lr_decay', numba.float64),
    ('delta', numba.float64),
    ('w', numba.float64[:]),
    ('next_u', numba.float64[:]),
    ('next_r', numba.int32[:])
])
class _DuelingBanditPolicy:
    def __init__(self, d, lr, lr_decay, delta, w, next_u, next_r):
        self.d = d
        self.lr = lr
        self.lr_decay = lr_decay
        self.delta = delta
        self.w = w
        self.next_u = next_u
        self.next_r = next_r

    def update(self, dataset, index, r, c, debug=False):
        # Count clicks
        count = 0
        p2 = 0
        for c_i in c:
            while p2 < self.next_r.shape[0] and self.next_r[p2] <= c_i:
                if self.next_r[p2] == c_i:
                    count += 1
                p2 += 1

        # Perform (projected) update if the new w is better
        if count > (len(c) - count):
            self.w += self.lr * self.next_u
            self.w /= np.linalg.norm(self.w)

        # Decay lr
        self.lr = self.lr * self.lr_decay

    def draw(self, x, debug=False):
        # Sample new model from unit sphere and store it
        u = np.random.randn(self.w.shape[0])
        u /= np.linalg.norm(u)
        self.next_u = u

        # Compute new weight vector
        next_w = self.w + self.delta * u
        next_w /= np.linalg.norm(next_w)

        # Compute rankings for old model (w) and new model (next_w)
        r1 = inorder(np.dot(x, self.w))
        r2 = inorder(np.dot(x, next_w))

        # Generate interleaved ranking
        int_ranking, t1, t2 = tdi(r1, r2)

        # Store which ranks would result in an update to the model
        self.next_r = t2

        # Return ranking
        return int_ranking

    def max(self, x):
        s = np.dot(x, self.w)
        return inorder(s)


@numba.njit(nogil=True)
def tdi(r1, r2):
    # Set pointers to 0 for the start of both rankings r1 and r2
    c_1 = 0
    c_2 = 0

    # Create teams and an interleaved ranking for output
    int_r = np.zeros(r1.shape[0] + r2.shape[0], dtype=np.int32)
    int_size = 0
    int_set = set()
    t1 = np.zeros(r1.shape[0] + r2.shape[0], dtype=np.int32)
    t1_size = 0
    t2 = np.zeros(r1.shape[0] + r2.shape[0], dtype=np.int32)
    t2_size = 0

    # Main TDI algorithm
    while c_1 < len(r1) and c_2 < len(r2):
        if t1_size < t2_size or (t1_size == t2_size and np.random.rand() > 0.5):
            int_r[int_size] = r1[c_1]
            int_set.add(r1[c_1])
            t1[t1_size] = int_size
            t1_size += 1
            int_size += 1
            c_1 += 1
        else:
            int_r[int_size] = r2[c_2]
            int_set.add(r2[c_2])
            t2[t2_size] = int_size
            t2_size += 1
            int_size += 1
            c_2 += 1

        # Increment pointers until we reach a document not yet
        # in the final ranking
        while c_1 < len(r1) and r1[c_1] in int_set:
            c_1 += 1
        while c_2 < len(r2) and r2[c_2] in int_set:
            c_2 += 1

    # Return interleaved results and the two teams
    return int_r[0:int_size], t1[0:t1_size], t2[0:t2_size]


def __getstate(self):
    return {
        'd': self.d,
        'lr': self.lr,
        'lr_decay': self.lr_decay,
        'delta': self.delta,
        'w': self.w,
        'next_u': self.next_u,
        'next_r': self.next_r
    }

def __setstate(self, state):
    self.d = state['d']
    self.lr = state['lr']
    self.lr_decay = state['lr_decay']
    self.delta = state['delta']
    self.w = state['w']
    self.next_u = state['next_u']
    self.next_r = state['next_r']


def __reduce(self):
    return (DuelingBanditPolicy, (self.d, self.lr, self.lr_decay, self.delta,
                                  self.w, self.next_u, self.next_r))


def __deepcopy(self):
    return DuelingBanditPolicy(self.d, self.lr, self.lr_decay, self.delta,
                               np.copy(self.w), np.copy(self.next_u),
                               self.next_r.copy())


def DuelingBanditPolicy(d, lr, lr_decay=1.0, delta=0.1, w=None, next_u=None,
                        next_r=None):
    w = np.zeros(d) if w is None else w
    next_u = np.zeros(d, dtype=np.float64) if next_u is None else next_u
    next_r = np.array([], dtype=np.int32) if next_r is None else next_r
    out = _DuelingBanditPolicy(d, lr, lr_decay, delta, w, next_u, next_r)
    setattr(out.__class__, '__getstate__', __getstate)
    setattr(out.__class__, '__setstate__', __setstate)
    setattr(out.__class__, '__reduce__', __reduce)
    setattr(out.__class__, '__deepcopy__', __deepcopy)
    return out
