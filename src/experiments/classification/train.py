import logging
import numpy as np
import numba
import matplotlib
import json
from matplotlib import pyplot as plt
from scipy import stats as st
from joblib.memory import Memory
from argparse import ArgumentParser
from rulpy.pipeline import task, TaskExecutor
from experiments.classification.policies import create_policy
from experiments.classification.optimization import optimize
from experiments.classification.evaluation import evaluate
from experiments.classification.baseline import best_baseline
from experiments.classification.dataset import load_train, load_test
from experiments.util import rng_seed, get_evaluation_points, mkdir_if_not_exists, NumpyEncoder


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)s: %(message)s",
                        level=logging.INFO)
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=15)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("-o", "--output", type=str, required=True)
    cli_parser.add_argument("--cache", type=str, default="cache")
    cli_parser.add_argument("--iterations", type=int, default=1000000)
    cli_parser.add_argument("--evaluations", type=int, default=50)
    cli_parser.add_argument("--eval_scale", choices=('lin', 'log'), default='log')
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--strategy", type=str, default='epsgreedy')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.01)
    parser.add_argument("--label", type=str, default=None)

    # Read experiment configuration
    with open(args.config, 'rt') as f:
        lines = f.readlines()
        configs = [parser.parse_args(line.strip().split(" ")) for line in lines]

    # Run experiments in task executor
    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [run_experiment(config, args.dataset, args.repeats, args.iterations, args.evaluations, args.eval_scale) for config in configs]
    results = [r.result for r in results]
    
    # Write json results
    mkdir_if_not_exists(f"results/{args.output}.json")
    with open(f"results/{args.output}.json", "wt") as f:
        js_results = [{"result": result, "args": vars(config)} for result, config in zip(results, configs)]
        json.dump(js_results, f, cls=NumpyEncoder)
    
    # Print results
    for result, config in sorted(zip(results, configs), key=lambda e: e[0]['best']['conf'][-1][0], reverse=True):
        best_res = f"{result['best']['mean'][-1]:.4f} +/- {result['best']['std'][-1]:.4f} => {result['best']['conf'][-1][0]:.4f}"
        policy_res = f"{result['policy']['mean'][-1]:.4f} +/- {result['policy']['std'][-1]:.4f} => {result['policy']['conf'][-1][0]:.4f}"
        logging.info(f"{config.strategy} ({config.lr}) = {best_res} (deterministic)   {policy_res} (policy)")

    # Create plot
    fig, ax = plt.subplots()
    for config, result in zip(configs, results):
        label = f"{config.strategy} ({config.lr})" if config.label is None else config.label
        x = result['x']
        y = result['best']['mean']
        y_std = result['best']['std']
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.35)
    if args.eval_scale == 'log':
        ax.set_xscale('symlog')
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Reward $r \in [0, 1]$')
    ax.legend()
    mkdir_if_not_exists(f"plots/{args.output}.pdf")
    fig.savefig(f"plots/{args.output}.pdf")


@task(use_cache=True)
async def run_experiment(config, data, repeats, iterations, evaluations, eval_scale, seed_base=4200):

    # points to evaluate at
    points = get_evaluation_points(iterations, evaluations, eval_scale)
    
    # Evaluate at all points and all seeds
    results = []
    for seed in range(seed_base, seed_base + repeats):
        results.append(classification_run(config, data, points, seed))

    # Await results to finish computing
    results = [await r for r in results]

    # Combine results with different seeded repeats
    results = {
        "best": np.vstack([x["model"] for x in results]),
        "policy": np.vstack([x["explore"] for x in results])
    }

    # Compute aggregate statistics from results
    out = {
        k: {
            "mean": np.mean(results[k], axis=0),
            "std": np.std(results[k], axis=0),
            "conf": st.t.interval(0.95, results[k].shape[0] - 1, loc=np.mean(results[k], axis=0), scale=st.sem(results[k], axis=0)),
            "n": results[k].shape[0]
        }
        for k in results.keys()
    }
    out["x"] = points
    
    # Return results
    return out


@task(use_cache=True)
async def classification_run(config, data, points, seed):

    # Load train, test and policy
    train = load_train(data, seed)
    test = load_test(data, seed)
    policy = build_policy(config, data, seed)
    train, test, policy = await train, await test, await policy
    policy = policy.__deepcopy__()

    # Data structure to hold output results
    out = {
        'explore': np.zeros(len(points)),
        'model': np.zeros(len(points))
    }

    # Generate training indices and seed randomness
    prng = rng_seed(seed)
    indices = prng.randint(0, train.n, np.max(points))

    # Evaluate on point 0
    out['explore'][0], out['model'][0] = evaluate(test, policy)
    log_progress(0, points, out, policy, config, seed)

    # Train and evaluate at specified points
    for i in range(1, len(points)):
        start = points[i - 1]
        end = points[i]
        optimize(train, indices[start:end], policy)
        out['explore'][i], out['model'][i] = evaluate(test, policy)
        log_progress(i, points, out, policy, config, seed)
    
    return out


@task
async def build_policy(config, data, seed):
    train = load_train(data, seed)
    baseline = best_baseline(data, seed)
    train, baseline = await train, await baseline
    args = {'k': train.k, 'd': train.d, 'baseline': baseline}
    args.update(vars(config))
    return create_policy(**args)


def log_progress(index, points, out, policy, config, seed):
    bounds = ""
    if hasattr(policy, 'ucb_baseline') and hasattr(policy, 'lcb_w'):
        bounds = f" :: {policy.ucb_baseline:.6f} <> {policy.lcb_w:.6f}"
    logging.info(f"[{points[index]:7d}, {seed}] {config.strategy}: {out['explore'][index]:.4f} (stochastic) {out['model'][index]:.4f} (deterministic) {bounds}")


if __name__ == "__main__":
    main()
