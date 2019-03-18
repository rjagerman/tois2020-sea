import numba


@numba.njit(nogil=True)
def reward(x, y, a):
    r = 0.0
    if a == y:
        r = 1.0
    return r
