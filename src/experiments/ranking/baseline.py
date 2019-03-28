import logging
import numpy as np
import json
from joblib.memory import Memory
from scipy import stats as st
from argparse import ArgumentParser
from rulpy.pipeline.task_executor import task, TaskExecutor
from experiments.util import rng_seed
from experiments.ranking.dataset import load_test, load_train
from experiments.ranking.optimization import optimize_supervised
from experiments.ranking.evaluation import evaluate_fraction
from experiments.ranking.policies.online import OnlinePolicy


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
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=10)
    with open(args.conf, "rt") as f:
        configs = [parser.parse_args(line.strip().split(" ")) for line in f.readlines()]

    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [
            evaluate_config(args.dataset, conf.lr, conf.fraction, conf.epochs, args.repeats)
            for conf in configs
        ]
    results = [r.result for r in results]
    
    for config, result in sorted(zip(configs, results), key=lambda e: e[1]['conf'][0], reverse=True):
        logging.info(f"{args.dataset} baseline: {result['mean']:.5f} +/- {result['std']:.5f} => {result['conf'][0]:.5f}  (lr = {result['lr']})")


@task(use_cache=True)
async def evaluate_config(data, lr, fraction, epochs, repeats, seed_base=4200):
    results = [
        evaluate_baseline(data, lr, fraction, epochs, seed)
        for seed in range(seed_base, seed_base + repeats)
    ]
    results =  np.array([await r for r in results])
    return {
        'lr': lr,
        'mean': np.mean(results),
        'std': np.std(results),
        'n': results.shape[0],
        'conf': st.t.interval(0.95, results.shape[0] - 1, loc=np.mean(results), scale=st.sem(results))
    }


@task
async def evaluate_baseline(data, lr, fraction, epochs, seed):
    test = load_test(data, seed)
    baseline = train_baseline(data, lr, fraction, epochs, seed)
    test, baseline = await test, await baseline
    rng_seed(seed)
    ndcg_score, _ = evaluate_fraction(test, baseline, fraction)
    logging.info(f"[{seed}, {lr}] evaluation baseline: {ndcg_score:.4f}")
    return ndcg_score


@task
async def train_baseline(data, lr, fraction, epochs, seed):
    train = await load_train(data, seed)
    policy = OnlinePolicy(train.d, lr)
    baseline_size = int(fraction * train.size)
    prng = rng_seed(seed)
    indices = prng.permutation(train.size)[0:baseline_size]
    logging.info(f"[{seed}, {lr}] training baseline (size: {baseline_size})")
    optimize_supervised(train, indices, policy, lr, epochs)
    return policy


@task
async def best_baseline(data, seed):
    with open("conf/ranking/baselines.json", "rt") as f:
        baselines = json.load(f)
    return await train_baseline(data, seed=seed, **baselines[data])


if __name__ == "__main__":
    main()
