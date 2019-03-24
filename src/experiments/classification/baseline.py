import logging
import numpy as np
import json
from joblib.memory import Memory
from scipy import stats as st
from argparse import ArgumentParser
from rulpy.pipeline.task_executor import task, TaskExecutor
from experiments.classification.policies import EpsgreedyPolicy
from experiments.classification.optimization import optimize_supervised_hinge
from experiments.classification.dataset import load_train, load_test
from experiments.classification.evaluation import evaluate
from experiments.util import rng_seed


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)s: %(message)s",
                        level=logging.INFO)

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--conf", type=str, required=True)
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=15)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("--cache", type=str, default="cache")
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.05)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    with open(args.conf, "rt") as f:
        configs = [parser.parse_args(line.strip().split(" ")) for line in f.readlines()]

    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [
            evaluate_config(args.dataset, conf.lr, conf.fraction, conf.epochs, conf.eps, args.repeats)
            for conf in configs
        ]
    results = [r.result for r in results]
    
    for result in results:
        best = np.array([run['best'] for run in result['performance']])
        policy = np.array([run['policy'] for run in result['performance']])
        result['performance'] = {
            'best': {
                'mean': np.mean(best),
                'std': np.std(best, ddof=1),
                'conf': st.t.interval(0.95, len(best)-1, loc=np.mean(best), scale=st.sem(best))
            },
            'policy': {
                'mean': np.mean(policy),
                'std': np.std(policy, ddof=1),
                'conf': st.t.interval(0.95, len(policy)-1, loc=np.mean(policy), scale=st.sem(policy))
            }
        }


    for r in sorted(results, key=lambda e: e['performance']['policy']['conf'][0], reverse=True):
        policy = r['performance']['policy']
        best = r['performance']['best']
        logging.info(f"eps={r['eps']} lr={r['lr']} :: {policy['mean']:.5f} +/- {policy['std']:.5f} -> {policy['conf'][0]:.5f} (95% LCB)")


@task(use_cache=True)
async def evaluate_config(data, lr, fraction, epochs, eps, repeats):
    results = {
        'eps': eps,
        'lr': lr,
        'performance': [
            evaluate_baseline(data, lr, fraction, epochs, eps, seed)
            for seed in range(4200, 4200 + repeats)
        ]
    }
    results['performance'] = [await r for r in results['performance']]
    return results


@task(use_cache=True)
async def evaluate_baseline(data, lr, fraction, epochs, eps, seed):
    test = load_test(data, seed)
    baseline = train_baseline(data, lr, fraction, epochs, eps, seed)
    test, baseline = await test, await baseline
    rng_seed(seed)
    acc_policy, acc_best = evaluate(test, baseline)
    logging.info(f"[{seed}, {lr}, {eps}] evaluation baseline: {acc_policy:.4f} (stochastic) {acc_best:.4f} (deterministic)")
    return {'policy': acc_policy, 'best': acc_best}


@task(use_cache=True)
async def train_baseline(data, lr, fraction, epochs, eps, seed):
    train = await load_train(data)
    model = EpsgreedyPolicy(train.k, train.d, lr=lr, eps=eps)
    baseline_size = int(fraction * train.n)
    prng = rng_seed(seed)
    indices = prng.permutation(train.n)[0:baseline_size]
    logging.info(f"[{seed}, {lr}, {eps}] training baseline (size: {baseline_size})")
    optimize_supervised_hinge(train, indices, model, lr, epochs)
    return model


@task
async def best_baseline(data, seed):
    with open("conf/baselines.json", "rt") as f:
        baselines = json.load(f)
    return await train_baseline(data, seed=seed, **baselines[data])


if __name__ == "__main__":
    main()
