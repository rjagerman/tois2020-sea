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
from experiments.util import rng_seed, get_evaluation_points, mkdir_if_not_exists, NumpyEncoder
from experiments.ranking.dataset import load_test, load_train
from experiments.ranking.policies import create_policy
from experiments.ranking.evaluation import evaluate
from experiments.ranking.optimization import optimize
from experiments.ranking.baseline import best_baseline
from ltrpy.clicks.position import position_binarized_5, near_random_5
from ltrpy.clicks.cascading import perfect_5
from ltrpy.clicks.cascading import navigational_5
from ltrpy.clicks.cascading import informational_5


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)-23s: %(message)s",
                        level=logging.INFO)
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-b", "--behavior", choices=('position', 'perfect', 'navigational', 'informational', 'nearrandom'), default='position')
    cli_parser.add_argument("-r", "--repeats", type=int, default=15)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("-o", "--output", type=str, required=True)
    cli_parser.add_argument("--cache", type=str, default="cache")
    cli_parser.add_argument("--iterations", type=int, default=1000000)
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
        configs = [parser.parse_args(line.strip().split(" ")) for line in lines if not line.startswith("#")]

    # Run experiments in task executor
    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [run_experiment(config, args.dataset, args.behavior, args.repeats, args.iterations, args.evaluations, args.eval_scale) for config in configs]
    results = [r.result for r in results]

    # Write json results
    mkdir_if_not_exists(f"results/{args.output}.json")
    with open(f"results/{args.output}.json", "wt") as f:
        js_results = [{"result": result, "args": vars(config)} for result, config in zip(results, configs)]
        json.dump(js_results, f, cls=NumpyEncoder)

    # Print results
    for metric in ["learned", "deploy", "regret"]:
        bound = 1 if metric == 'regret' else 0
        reverse = False if metric == 'regret' else True
        for result, config in sorted(zip(results, configs), key=lambda e: e[0][metric]['conf'][bound][-1], reverse=reverse):
            logging.info(f"{args.dataset} {args.behavior} {config.strategy} ({config.lr}): {result[metric]['mean'][-1]:.5f} +/- {result[metric]['std'][-1]:.5f} => {result[metric]['conf'][bound][-1]:.5f}")

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
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.linspace(0.1, 1.0, 10))
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('ndcg@10')
    ax.legend()
    mkdir_if_not_exists(f"plots/{args.output}.pdf")
    fig.savefig(f"plots/{args.output}.pdf")


@task(use_cache=True)
async def run_experiment(config, data, behavior, repeats, iterations, evaluations, eval_scale, seed_base=4200):

    # points to evaluate at
    points = get_evaluation_points(iterations, evaluations, eval_scale)

    # Evaluate at all points and all seeds
    results = []
    for seed in range(seed_base, seed_base + repeats):
        results.append(ranking_run(config, data, behavior, points, seed))

    # Await results to finish computing
    final_results = [await r for r in results]
    results = {
        "deploy": np.vstack([r["deploy"] for r in final_results]),
        "learned": np.vstack([r["learned"] for r in final_results]),
        "regret": np.vstack([r["regret"] for r in final_results])
    }
    if "ucb_b" in final_results[0].keys() and "lcb_w" in final_results[0].keys():
        results["ucb_b"] = np.vstack([r["ucb_b"] for r in final_results])
        results["lcb_w"] = np.vstack([r["lcb_w"] for r in final_results])

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
async def ranking_run(config, data, behavior, points, seed):

    # Load train, test and policy
    train = load_train(data, seed)
    test = load_test(data, seed)
    baseline = best_baseline(data, seed)
    train, test, baseline = await train, await test, await baseline

    # Data structure to hold output results
    out = {
        'deploy': np.zeros(len(points)),
        'learned': np.zeros(len(points)),
        'regret': np.zeros(len(points))
    }

    # Seed randomness
    prng = rng_seed(seed)

    # Build policy
    args = {'d': train.d, 'pairs': train.pairs, 'baseline': baseline.__deepcopy__()}
    args.update(vars(config))
    if behavior in ['perfect']:
        args['eta'] = 0.0
    else:
        args['eta'] = 1.0
    if not config.cold:
        args['w'] = np.copy(baseline.w)
    policy = create_policy(**args)

    if hasattr(policy, 'ucb_baseline') and hasattr(policy, 'lcb_w'):
        out['ucb_b'] = np.zeros(len(points))
        out['lcb_w'] = np.zeros(len(points))

    # Build behavior model
    click_model = build_click_model(behavior)

    # Generate training indices and seed randomness
    indices = prng.randint(0, train.size, np.max(points))

    # Evaluate on point 0
    out['deploy'][0], out['learned'][0] = evaluate(test, policy)
    log_progress(0, points, seed, data, behavior, config, out, policy)

    # Train and evaluate at specified points
    for i in range(1, len(points)):
        start = points[i - 1]
        end = points[i]
        out['regret'][i] = out['regret'][i - 1] + optimize(train, indices[start:end], policy, click_model)
        out['deploy'][i], out['learned'][i] = evaluate(test, policy)
        if hasattr(policy, 'ucb_baseline') and hasattr(policy, 'lcb_w'):
            out['ucb_b'], out['lcb_w'] = policy.ucb_baseline, policy.lcb_w
        log_progress(i, points, seed, data, behavior, config, out, policy)

    return out


def log_progress(i, points, seed, data, behavior, config, out, policy):
    bounds = ""
    if hasattr(policy, 'ucb_baseline') and hasattr(policy, 'lcb_w'):
        bounds = f" :: {policy.lcb_w:.6f} ?> {policy.ucb_baseline:.6f}"
    logging.info(f"[{seed}, {points[i]:7d}] {data} {behavior} {config.strategy} ({config.lr}): {out['learned'][i]:.5f} (learned) {out['deploy'][i]:.5f} (deploy){bounds}")


def build_click_model(behavior):
    return {
        'position': position_binarized_5,
        'perfect': perfect_5,
        'navigational': navigational_5,
        'informational': informational_5,
        'nearrandom': near_random_5
    }[behavior](10)


if __name__ == "__main__":
    main()
