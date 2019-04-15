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
from experiments.classification.baseline import best_baseline, statistical_baseline
from experiments.classification.dataset import load_train, load_test
from experiments.util import rng_seed, get_evaluation_points, mkdir_if_not_exists, NumpyEncoder


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)-23s: %(message)s",
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
    parser.add_argument("--cold", action='store_true')
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
    for metric in ['learned', 'deploy', 'regret']:
        bound = 1 if metric == 'regret' else 0
        reverse = False if metric == 'regret' else True
        for result, config in sorted(zip(results, configs), key=lambda e: e[0][metric]['conf'][bound][-1], reverse=reverse):
            tune_p = config.lr
            if config.strategy in ["ucb", "thompson"]:
                tune_p = config.alpha
            logging.info(f"{args.dataset} {config.strategy} ({tune_p}) = {metric}: {result[metric]['mean'][-1]:.4f} +/- {result[metric]['std'][-1]:.4f} => {result[metric]['conf'][bound][-1]:.4f}")

    # Create plot
    fig, ax = plt.subplots()
    for config, result in zip(configs, results):
        label = f"{config.strategy} ({config.lr})" if config.label is None else config.label
        x = result['x']
        y = result['learned']['mean']
        y_std = result['learned']['std']
        ax.plot(x, y, label=label)
        ax.fill_between(x, y - y_std, y + y_std, alpha=0.35)
    if args.eval_scale == 'log':
        ax.set_xscale('symlog')
        locmin = matplotlib.ticker.SymmetricalLogLocator(base=10.0, subs=np.linspace(1.0, 10.0, 10), linthresh=1.0) 
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        scalar_formatter = matplotlib.ticker.ScalarFormatter()
        log_formatter = matplotlib.ticker.LogFormatterSciNotation(linthresh=1.0)
        def smart_formatter(x, p):
            if x in [1.0, 0.0]:
                return scalar_formatter.format_data(x)
            else:
                return log_formatter.format_data(x)
        func_formatter = matplotlib.ticker.FuncFormatter(smart_formatter)
        ax.xaxis.set_major_formatter(func_formatter)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Reward $r \in [0, 1]$')
    #ax.set_ylim([0.0, 1.0])
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
        "learned": np.vstack([x["learned"] for x in results]),
        "deploy": np.vstack([x["deploy"] for x in results]),
        "regret": np.vstack([x["regret"] for x in results])
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
        'deploy': np.zeros(len(points)),
        'learned': np.zeros(len(points)),
        'regret': np.zeros(len(points))
    }

    # Generate training indices and seed randomness
    prng = rng_seed(seed)
    indices = prng.randint(0, train.n, np.max(points))

    # Evaluate on point 0
    out['deploy'][0], out['learned'][0] = evaluate(test, policy)
    out['regret'][0] = 0.0
    log_progress(0, points, data, out, policy, config, seed)

    # Train and evaluate at specified points
    for i in range(1, len(points)):
        start = points[i - 1]
        end = points[i]
        out['regret'][i] = out['regret'][i - 1] + optimize(train, indices[start:end], policy)
        out['deploy'][i], out['learned'][i] = evaluate(test, policy)
        log_progress(i, points, data, out, policy, config, seed)
    
    return out


@task
async def build_policy(config, data, seed):
    train = load_train(data, seed)
    if config.strategy in ['ucb', 'thompson']:
        baseline = statistical_baseline(data, seed, config.strategy)
    else:
        baseline = best_baseline(data, seed)
    train, baseline = await train, await baseline
    if not config.cold and config.strategy in ['ucb', 'thompson']:
        return baseline.__deepcopy__()
    args = {'k': train.k, 'd': train.d, 'n': train.n, 'baseline': baseline}
    args.update(vars(config))
    if not config.cold:
        args['w'] = np.copy(baseline.w)
    return create_policy(**args)


def log_progress(index, points, data, out, policy, config, seed):
    bounds = ""
    if hasattr(policy, 'ucb_baseline') and hasattr(policy, 'lcb_w'):
        bounds = f" :: {policy.lcb_w:.6f} ?> {policy.ucb_baseline:.6f}"
    tune = config.l2 if config.strategy in ["ucb", "thompson"] else config.lr
    logging.info(f"[{seed}, {points[index]:7d}] {data} {config.strategy} ({tune}): {out['deploy'][index]:.4f} (deploy) {out['learned'][index]:.4f} (learned) {bounds}")


if __name__ == "__main__":
    main()
