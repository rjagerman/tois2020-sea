import logging
import numpy as np
import numba
from scipy import stats as st
from joblib.memory import Memory
from argparse import ArgumentParser
from experiments.classification import dataset
from rulpy.pipeline import task, TaskExecutor
from experiments.classification.policies import create_policy, policy_from_model
from experiments.classification.optimization import optimize
from experiments.classification.evaluation import evaluate
from experiments.classification.baseline import train_baseline
from experiments.classification.dataset import load_train, load_test
from experiments.util import rng_seed, get_evaluation_points


def main():
    logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)s: %(message)s",
                        level=logging.INFO)
    cli_parser = ArgumentParser()
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    cli_parser.add_argument("-d", "--dataset", type=str, required=True)
    cli_parser.add_argument("-r", "--repeats", type=int, default=15)
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("--cache", type=str, default="cache")
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--evaluations", type=int, default=50)
    parser.add_argument("--eval_scale", choices=('lin', 'log'), default='lin')
    parser.add_argument("--strategy", type=str, default='epsgreedy')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--baseline_fraction", type=float, default=0.01)
    parser.add_argument("--baseline_lr", type=float, default=0.01)
    parser.add_argument("--baseline_tau", type=float, default=1.0)
    parser.add_argument("--baseline_epochs", type=int, default=50)

    with open(args.config, 'rt') as f:
        lines = f.readlines()
        configs = [parser.parse_args(line.strip().split(" ")) for line in lines]

    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [run_experiment(config, args.dataset, args.repeats) for config in configs]
    
    for result, config in sorted(zip(results, configs), key=lambda e: e[0].result['best']['conf'][-1][0], reverse=True):
        best_res = f"{result.result['best']['mean'][-1]:.4f} +/- {result.result['best']['std'][-1]:.4f} => {result.result['best']['conf'][-1][0]:.4f}"
        policy_res = f"{result.result['policy']['mean'][-1]:.4f} +/- {result.result['policy']['std'][-1]:.4f} => {result.result['policy']['conf'][-1][0]:.4f}"
        logging.info(f"{config.strategy} ({config.lr}) = {best_res} (deterministic)   {policy_res} (policy)")


@task(use_cache=True)
async def run_experiment(config, dataset, repeats, seed_base=4200):
    
    # Load training data
    train = await load_train(dataset)

    # points to evaluate at
    points = get_evaluation_points(config.iterations, config.evaluations,
                                   config.eval_scale)
    
    # Evaluate at all points and all seeds
    results = []
    for index in range(len(points)):
        run = []
        for seed in range(seed_base, seed_base + repeats):
            prng = rng_seed(seed)
            indices = prng.randint(0, train.n, np.max(points))
            run.append(evaluate_model(config, dataset, indices, points, index, seed))
        results.append(run)
    
    # Await results and compute mean/std/n of results
    out = {'best': {'mean': [], 'std': [], 'conf': [], 'n': []}, 'policy': {'mean': [], 'std': [], 'conf': [], 'n': []}, 'x': points}
    for t in ['best', 'policy']:
        for runs in results:
            arr = np.array([(await r)[t] for r in runs])
            out[t]['mean'].append(np.mean(arr))
            out[t]['std'].append(np.std(arr))
            out[t]['conf'].append(st.t.interval(0.95, len(arr)-1, loc=np.mean(arr), scale=st.sem(arr)))
            out[t]['n'].append(arr.shape[0])

    # Await for all results to compute, then return it
    return out


@task
async def build_policy(config, dataset, seed):
    train = load_train(dataset)
    baseline = train_baseline(dataset, config.baseline_lr, config.baseline_fraction,
                              config.baseline_epochs, config.baseline_tau, seed)
    train, baseline = await train, await baseline

    return create_policy(train.d, train.k, config.strategy, config.lr, config.l2,
                         config.eps, config.tau, config.alpha, config.cap, baseline)


@task(use_cache=True)
async def evaluate_model(config, dataset, indices, points, index, seed):
    # Load test data, model and test_policy
    train = load_train(dataset)
    test = load_test(dataset)
    model = train_model(config, dataset, indices, points, index, seed)
    policy = build_policy(config, dataset, seed)

    # Wait for sub tasks to finish
    train, test, model, policy = await train, await test, await model, await policy

    # Evaluate results with the test policy
    rng_seed(seed)
    acc_policy, acc_best = evaluate(test, model, policy)
    logging.info(f"[{points[index]:7d}] {config.strategy}: {acc_policy:.4f} (stochastic) {acc_best:.4f} (deterministic)")
    return {
        'policy': acc_policy,
        'best': acc_best
    }


@task(use_cache=True)
async def train_model(config, dataset, indices, points, index, seed):
    # Load train data
    train = load_train(dataset)
    policy = build_policy(config, dataset, seed)
    train, policy = await train, await policy

    # If we are at iteration 0, just return initialized model
    if index == 0:
        return policy.create()
    
    model = await train_model(config, dataset, indices, points, index - 1, seed)
    model = policy.copy(model)

    # Train actual model on the data
    rng_seed(seed)
    indices = indices[points[index-1]:points[index]]
    optimize(train, indices, model, policy)

    # Return the trained model
    return model


if __name__ == "__main__":
    main()
