import logging
import numpy as np
import numba
import matplotlib
import json
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy import stats as st
from skopt.space import Real, Space
from joblib.memory import Memory
from argparse import ArgumentParser
from rulpy.pipeline import task, TaskExecutor
from experiments.classification.train import run_experiment
from experiments.util import LogGridOptimizer, NumpyEncoder


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)-23s: %(message)s",
                        level=logging.INFO)
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=15)
    cli_parser.add_argument("-a", "--attempts", type=int, default=30)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("--cache", type=str, default="cache")
    cli_parser.add_argument("--iterations", type=int, default=1_000_000)
    args = cli_parser.parse_args()

    # Search parameters
    parser = ArgumentParser()
    parser.add_argument("--strategy", type=str, default='epsgreedy')
    parser.add_argument("--cold", action='store_true')
    parser.add_argument("--x0_min", type=float, default=1e-10)
    parser.add_argument("--x0_max", type=float, default=10.0)
    parser.add_argument("--x0_prior", type=str, default='log-uniform')
    parser.add_argument("--x1_min", type=float, default=1e-10)
    parser.add_argument("--x1_max", type=float, default=1000.0)
    parser.add_argument("--x1_prior", type=str, default='log-uniform')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.1)
    
    # Read experiment configuration
    with open(args.config, 'rt') as f:
        lines = f.readlines()
        configs = [parser.parse_args(line.strip().split(" ")) for line in lines]

    # Run experiments in task executor
    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        targets = []
        for config in configs:
            space = Space([
                Real(low=config.x0_min, high=config.x0_max, prior=config.x0_prior),
                Real(low=config.x1_min, high=config.x1_max, prior=config.x1_prior)
            ])
            maximize = True if config.strategy == 'ips' else False
            hyperopt = LogGridOptimizer(target_fn, space, maximize=maximize, max_parallel=5, kwargs={
                "config": config,
                "data": args.dataset,
                "repeats": args.repeats,
                "iterations": args.iterations,
                "seed_base": 4200
            }, bases=[1, 3])
            targets.append(hyperopt)
        results = [target.optimize(args.attempts if args.attempts != -1 else hyperopt.nr_max_attempts) for target in targets]
    results = [r.result for r in results]

    for target, result in zip(targets, results):
        logging.info(f"{target.kwargs['config'].strategy} == LR:{result[0][0]}, L2:{result[0][1]} -> performance (95% CI) = {result[1]}")


@task(use_cache=False)
async def target_fn(x0, x1, config, data, repeats, iterations, seed_base, call_uid=None):
    new_config = deepcopy(config)
    if new_config.strategy in ['ucb', 'thompson']:
        new_config.alpha = x0
    else:
        new_config.lr = x0
    new_config.l2 = x1
    output = await run_experiment(new_config, data, repeats, iterations, 2, 'lin', seed_base, vali=0.1)
    if new_config.strategy in ['ips']:
        return output['learned']['conf'][0][-1]
    return output['test_regret']['conf'][1][-1]


if __name__ == "__main__":
    main()
