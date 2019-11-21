import logging
import numpy as np
import numba
import matplotlib
import json
from collections import defaultdict
from copy import deepcopy
from matplotlib import pyplot as plt
from scipy import stats as st
from joblib.memory import Memory
from argparse import ArgumentParser
from rulpy.pipeline import task, TaskExecutor
from experiments.util import rng_seed, get_evaluation_points, mkdir_if_not_exists, NumpyEncoder
from experiments.ranking.train import run_experiment


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)-23s: %(message)s",
                        level=logging.INFO)
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    cli_parser.add_argument("-b", "--behavior", type=str, default="perfect")
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=5)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("--cache", type=str, default="cache")
    cli_parser.add_argument("--iterations", type=int, default=10000000)
    cli_parser.add_argument("--evaluations", type=int, default=50)
    cli_parser.add_argument("--eval_scale", choices=('lin', 'log'), default='log')
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--strategy", type=str, default='online')
    parser.add_argument("--cold", action='store_true')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--lr_decay", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--cap", type=float, default=0.01)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--label", type=str, default=None)

    # Read experiment configuration
    with open(args.config, 'rt') as f:
        lines = f.readlines()
        configs = [parser.parse_args(line.strip().split(" "))
                   for line in lines if not line.startswith("#")]
    default_args = parser.parse_args([])


    # Run experiments in task executor
    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [search_around(3, default_args, config, args.dataset, args.behavior, args.repeats, args.iterations) for config in configs]
    results = [r.result for r in results]

    # Print results
    logging.info("best settings")
    for result in results:
        logging.info(config_string(result, default_args))


def config_string(new_conf, default_conf, always_include=["strategy", "lr", "l2"]):
    out = ""
    for a in vars(default_conf):
        if a in always_include or getattr(new_conf, a) != getattr(default_conf, a):
            out += f" --{a} {getattr(new_conf, a)}"
    return out


def get_spectrum(minimum=-20.0, maximum=20.0, include_zero=True):
    base1_ls = np.logspace(minimum, maximum,num=maximum - minimum + 1)
    base3_ls = np.logspace(minimum, maximum,num=maximum - minimum + 1) * 3
    full_spectrum = np.sort(np.hstack([np.array([0.0]), base1_ls, base3_ls]))
    if include_zero:
        return full_spectrum
    else:
        return full_spectrum[1:]


def arg_closest(spectrum, value):
    return np.argmin(np.abs(value - spectrum))


def create_config(config, x0_name, x0):
    new_config = deepcopy(config)
    setattr(new_config, x0_name, x0)
    return new_config


def get_param_search_info(config):
    x0_name = "lr"
    return x0_name, getattr(config, x0_name)


@task(use_cache=False)
async def search_around(n, default_args, config, dataset, behavior, repeats, iterations):
    # Get parameters to search
    spectrum = get_spectrum()
    x0_name, x0 = get_param_search_info(config)
    ax0 = arg_closest(spectrum, x0)

    # Run parameter sweep
    result = defaultdict(dict)
    configs = {}
    for i in range(ax0 - n, ax0 + n + 1):
        configs[i] = create_config(config, x0_name, spectrum[i])
        result[i] = target_fn(configs[i], dataset, behavior, repeats,
                              iterations, 4200)

    # Wait for all jobs to finish
    for i in range(ax0 - n, ax0 + n + 1):
        result[i] = await result[i]

    # Print important results and store best config
    best = (None, np.inf)
    for i in range(ax0 - n, ax0 + n + 1):
        if result[i] < best[1]:
            best = (configs[i], result[i])
        out = f"{result[i]:.2f} ::: {config_string(configs[i], default_args)}"
        logging.info(out)

    return best[0]


@task(use_cache=False)
async def target_fn(config, data, behavior, repeats, iterations, seed_base):
    output = await run_experiment(config, data, behavior, repeats, iterations, 2,
                                  'lin', seed_base, vali=0.1)
    if config.strategy in ['ips']:
        return -output['learned']['conf'][0][-1]
    return output['regret']['conf'][1][-1]


if __name__ == "__main__":
    main()
