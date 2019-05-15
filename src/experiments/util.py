import numpy as np
import numba
import os
import json
import dlib
from rulpy.pipeline.task_executor import task
from skopt.space import Real, Integer, Categorical, Space
from threading import Semaphore


@numba.njit(nogil=True)
def _numba_rng_seed(seed):
    np.random.seed(seed)


def rng_seed(seed=None):
    np.random.seed(seed)
    _numba_rng_seed(seed)
    return np.random.RandomState(seed)


def get_evaluation_points(iterations, evaluations, scale):
    if scale == 'log':
        evaluations = 1 + int(np.log10(iterations)) * int(evaluations / np.log10(iterations))
        return np.unique(np.concatenate((np.zeros(1, dtype=np.int32), np.geomspace(1, iterations, evaluations, dtype=np.int32))))
    elif scale == 'lin':
        return np.concatenate((np.arange(0, iterations, iterations / evaluations, dtype=np.int32), np.array([iterations], dtype=np.int32)))
    else:
        return np.array([iterations], dtype=np.int32)


def mkdir_if_not_exists(path):
    directory = os.path.dirname(path)
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


@numba.njit(nogil=True)
def mpeb_bound(n, confidence, var, maximum):
    C = 1.0 - confidence
    out = (7 * maximum * np.log(2.0 / C)) / (3 * (n - 1))
    out += (1.0 / n) * np.sqrt(np.log(2.0 / C) / (n - 1) * var * (2.0 * n**2))
    return out


@numba.njit(nogil=True)
def ch_bound(n, confidence, maximum):
    C = 1.0 - confidence
    return maximum * np.sqrt(np.log(1.0 / C) / (2*n))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@task
async def hypermax(n, target_fn, space):
    """
    Hyper parameter optimization function
    """
    lowers = [x[0] for x in space.transformed_bounds]
    uppers = [x[1] for x in space.transformed_bounds]
    constraints = dlib.function_spec(lowers, uppers)
    solver = dlib.global_function_search(constraints)

    @task(use_cache=False)
    async def _next_fn_call(index):
        solver_call = solver.get_next_x()
        solver_call.set(await target_fn(*solver_call.x))

    output = [_next_fn_call(index) for index in range(n)]
    output = [await o for o in output]

    best_params, best_score, _ = solver.get_best_function_eval()
    best_params = space.inverse_transform(np.array([best_params]))[0]

    return best_params, best_score


class HyperOptimizer():
    def __init__(self, target_fn, space, maximize=True, max_parallel=5, kwargs={}):
        self.target_fn = target_fn
        self.space = space
        lowers = [x[0] for x in space.transformed_bounds]
        uppers = [x[1] for x in space.transformed_bounds]
        constraints = dlib.function_spec(lowers, uppers)
        self.solver = dlib.global_function_search(constraints)
        self.maximize = maximize
        self._can_run = Semaphore(max_parallel)
        self.kwargs = kwargs

    @task(use_cache=False)
    async def optimize(self, attempts):
        output = [self._next_run(index) for index in range(attempts)]
        output = [await o for o in output]
        best_params, best_score, _ = self.solver.get_best_function_eval()
        best_params = self.space.inverse_transform(np.array([best_params]))[0]
        if not self.maximize:
            best_score = -best_score
        return best_params, best_score
    
    @task(use_cache=False)
    async def _next_run(self, index):
        while not self._can_run.acquire(blocking=False):
            await DoNothing()
        try:
            point = self.solver.get_next_x()
            next_x = self.space.inverse_transform(np.array([point.x]))[0]
            result = await self.target_fn(*next_x, **(self.kwargs))
            if not self.maximize:
                result *= -1
            point.set(result)
        finally:
            self._can_run.release()

class DoNothing():
    def __await__(self):
        yield
        return True
