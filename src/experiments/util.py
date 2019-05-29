import numpy as np
import numba
import os
import json
import dlib
import logging
from rulpy.pipeline.task_executor import task
from skopt.space import Real, Integer, Categorical, Space
from threading import Semaphore, Lock


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


class HyperOptimizer():
    def __init__(self, target_fn, space, maximize=True, max_parallel=5, kwargs={}):
        self.target_fn = target_fn
        self.space = space
        self.solver = None
        self.maximize = maximize
        self._can_run = Semaphore(max_parallel)
        self._update_lock = Lock()
        self.kwargs = kwargs
        self._call_uid = id(self)

    @task(use_cache=False)
    async def optimize(self, attempts):
        logging.info(f"Starting hyper parameter search with {attempts} attempts")
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
            result = await self.target_fn(*next_x, **(self.kwargs), call_uid=self._call_uid)
            if not self.maximize:
                result *= -1
            with self._update_lock:
                point.set(result)
        finally:
            self._can_run.release()


class MaxLIPO_TR_Optimizer(HyperOptimizer):
    def __init__(self, target_fn, space, maximize=True, max_parallel=5, kwargs={}):
        super().__init__(target_fn, space, maximize, max_parallel, kwargs)
        lowers = [x[0] for x in space.transformed_bounds]
        uppers = [x[1] for x in space.transformed_bounds]
        constraints = dlib.function_spec(lowers, uppers)
        self.solver = dlib.global_function_search(constraints)


class LogGridOptimizer(HyperOptimizer):
    def __init__(self, target_fn, space, maximize=True, max_parallel=5, kwargs={}, bases=[1], seed=4200):
        super().__init__(target_fn, space, maximize, max_parallel, kwargs)
        lowers = [x[0] for x in space.transformed_bounds]
        uppers = [x[1] for x in space.transformed_bounds]
        self.solver = _LogGridSolver(lowers, uppers, bases, seed)

    @property
    def nr_max_attempts(self):
        return len(self.solver.grid_options)


class DoNothing():
    def __await__(self):
        yield
        return True


class _LogGridSolver():
    def __init__(self, lowers, uppers, bases=[1], seed=4200):
        self._t = 0
        dimensions = [np.arange(lowers[i], uppers[i] + 1) for i in range(len(uppers))]
        self.grid_options = np.stack(np.meshgrid(*dimensions), -1).reshape(-1, len(dimensions))
        self.grid_options = np.vstack([np.log10(base * (10 ** self.grid_options)) for base in bases])
        prng = np.random.RandomState(seed=seed)
        prng.shuffle(self.grid_options)
        self.grid_options = [
            GridPoint(self.grid_options[i, :], self)
            for i in range(self.grid_options.shape[0])
        ]
        self.best = self.grid_options[0]
        self._update_lock = Lock()
    
    def get_next_x(self):
        x = self.grid_options[self._t]
        self._t = (self._t + 1) % len(self.grid_options)
        return x

    def _update_point(self, point):
        with self._update_lock:
            if self.best._v is None or point._v > self.best._v:
                self.best = point

    def get_best_function_eval(self):
        return self.best.x, self.best._v, self.best

        
class GridPoint():
    def __init__(self, x, solver):
        self.x = x
        self._v = None
        self._solver = solver

    def set(self, result):
        self._v = result
        self._solver._update_point(self)
