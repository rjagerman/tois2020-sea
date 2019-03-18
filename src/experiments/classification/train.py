import logging
import numpy as np
import numba
import experiments.classification.dataset as dataset
from rulpy.pipeline import task, TaskExecutor
from rulpy.linalg import regression
from rulpy.pipeline import task, TaskExecutor
from experiments.classification.policies import create_policy, policy_from_model
from experiments.util import rng_seed
from collections import namedtuple


ExperimentConfig = namedtuple('ExperimentConfig', [
    'train_path',
    'test_path',
    'seed',
    'strategy',
    'baseline',
    'lr',
    'l2',
    'eps',
    'tau',
    'alpha'
])


def get_evaluation_points(iterations, evaluations, scale):
    if scale == 'log':
        evaluations = 1 + int(np.log10(iterations)) * int(evaluations / np.log10(iterations))
        return np.unique(np.concatenate((np.zeros(1, dtype=np.int32), np.geomspace(1, iterations, evaluations, dtype=np.int32))))
    elif scale == 'lin':
        return np.concatenate((np.arange(0, iterations, iterations / evaluations, dtype=np.int32), np.array([iterations], dtype=np.int32)))
    else:
        return np.array([iterations], dtype=np.int32)


@task(use_cache=True)
async def run_experiment(config):
    
    # Load training data
    train = await load_data(config.train_path)

    # points to evaluate at
    points = get_evaluation_points(config.iterations, config.evaluations,
                                   config.eval_scale)
    
    # Seed randomness and get data indices to train on
    rng_seed(config.seed)
    indices = np.random.randint(0, train.n, np.max(points))

    # Get results
    results = [
        evaluate_model(config, indices, points, index) for index in range(len(points))
    ]

    # Await for all results to compute, then return it
    return {
        'best': np.array([(await r)['best'] for r in results]),
        'policy': np.array([(await r)['policy'] for r in results]),
        'x': points
    }


@task
async def build_policy(config):
    train = load_data(config.train_path)
    baseline = train_baseline(config.train_path, config.baseline_lr, config.baseline_fraction,
                              config.baseline_epochs, config.baseline_tau, config.seed)
    train, baseline = await train, await baseline

    return create_policy(train.d, train.k, config.strategy, config.lr, config.l2,
                         config.eps, config.tau, config.alpha, config.cap, baseline)


@task(use_cache=True)
async def load_data(file_path, min_size=0):
    logging.info(f"Loading data set from {file_path}")
    return dataset.load(file_path, min_size)


@task(use_cache=True)
async def evaluate_model(config, indices, points, index):
    # Load test data, model and test_policy
    train = load_data(config.train_path)
    model = train_model(config, indices, points, index)
    policy = build_policy(config)

    # Wait for sub tasks to finish
    train, model, policy = await train, await model, await policy
    test = await load_data(config.test_path, train.d)

    # Evaluate results with the test policy
    rng_seed(config.seed)
    acc_policy, acc_best = evaluate(test, model, policy)
    logging.info(f"[{points[index]:7d}] {config.strategy}: {acc_policy:.4f} (stochastic) {acc_best:.4f} (deterministic)")
    return {
        'policy': acc_policy,
        'best': acc_best
    }


@task(use_cache=True)
async def train_model(config, indices, points, index):
    # Load train data
    train = load_data(config.train_path)
    policy = build_policy(config)
    train, policy = await train, await policy

    # If we are at iteration 0, just return initialized model
    if index == 0:
        return policy.create()
    
    model = await train_model(config, indices, points, index - 1)
    model = policy.copy(model)

    # Train actual model on the data
    rng_seed(config.seed)
    indices = indices[points[index-1]:points[index]]
    optimize(train, indices, model, policy)

    # Return the trained model
    return model

@task(use_cache=True)
async def evaluate_baseline(train_path, test_path, lr, fraction, epochs, tau,
                            seed):
    baseline = train_baseline(train_path, lr, fraction, epochs, tau, seed)
    train = load_data(train_path)
    baseline, train = await baseline, await train
    test = await load_data(test_path, min_size=train.d)
    policy = policy_from_model(baseline)
    rng_seed(seed)
    acc_policy, acc_best = evaluate(test, baseline, policy)
    return {'policy': acc_policy, 'best': acc_best}


@task(use_cache=True)
async def train_baseline(train_path, baseline_lr, baseline_fraction,
                         baseline_epochs, baseline_tau, seed):
    train = await load_data(train_path)
    logging.info(f"Training baseline model")
    rng_seed(seed)
    baseline_size = int(baseline_fraction * train.n)
    indices = np.random.permutation(train.n)[0:baseline_size]
    policy = create_policy(train.d, train.k, 'boltzmann', tau=baseline_tau)
    model = policy.create()
    optimize_supervised_hinge(train, indices, model, baseline_lr, baseline_epochs)
    return model


@numba.njit(nogil=True)
def optimize_supervised_hinge(train, indices, model, lr, epochs):
    w = model.w
    for e in range(epochs):
        np.random.shuffle(indices)
        for i in indices:
            grad = np.zeros((train.k, train.d))
            x, y = dataset.get(train, i)
            s = np.dot(w, x)
            for j in range(train.k):
                if j != y:
                    if s[j] - s[y] + 1 > 0.0:
                        grad[y, :] -= x
                        grad[j, :] += x
            w -= lr * grad


@numba.njit(nogil=True)
def optimize(train, indices, model, policy):
    for index in indices:
        x, y = dataset.get(train, index)
        a = policy.draw(model, x)
        r = reward(x, y, a)
        policy.update(model, x, a, r)


@numba.njit(nogil=True)
def evaluate(test_data, model, policy):
    cum_r_policy = 0.0
    cum_r_best = 0.0
    for i in range(test_data.n):
        x, y = dataset.get(test_data, i)
        a_policy = policy.draw(model, x)
        a_best = policy.best(model, x)
        r_policy = reward(x, y, a_policy)
        r_best = reward(x, y, a_best)
        cum_r_policy += r_policy
        cum_r_best += r_best
    return cum_r_policy / test_data.n, cum_r_best / test_data.n


@numba.njit(nogil=True)
def reward(x, y, a):
    r = 0.0
    if a == y:
        r = 1.0
    return r
