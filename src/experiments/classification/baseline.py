import logging
import numpy as np
from joblib.memory import Memory
from scipy import stats as st
from argparse import ArgumentParser
from rulpy.pipeline.task_executor import task, TaskExecutor
from experiments.classification.policies import create_policy, policy_from_model
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
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    with open(args.conf, "rt") as f:
        configs = [parser.parse_args(line.strip().split(" ")) for line in f.readlines()]

    with TaskExecutor(max_workers=args.parallel, memory=Memory(args.cache, compress=6)):
        results = [
            {
                'tau': conf.tau,
                'lr': conf.lr,
                'performance': [
                    evaluate_baseline(args.dataset, conf.lr, conf.fraction, conf.epochs, conf.tau, seed)
                    for seed in range(4200, 4200 + args.repeats)
                ]
            }
            for conf in configs
        ]
    
    for result in results:
        best = np.array([run.result['best'] for run in result['performance']])
        policy = np.array([run.result['policy'] for run in result['performance']])
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
        logging.info(f"tau={r['tau']} lr={r['lr']} :: {policy['mean']:.5f} +/- {policy['std']:.5f} -> {policy['conf'][0]:.5f} (95% LCB)")


@task(use_cache=True)
async def evaluate_baseline(dataset, lr, fraction, epochs, tau, seed):
    baseline = train_baseline(dataset, lr, fraction, epochs, tau, seed)
    test = load_test(dataset)
    baseline, test = await baseline, await test
    policy = policy_from_model(baseline)
    rng_seed(seed)
    acc_policy, acc_best = evaluate(test, baseline, policy)
    logging.info(f"[{seed}, {lr}, {tau}] baseline: {acc_policy:.4f} (stochastic) {acc_best:.4f} (deterministic)")
    return {'policy': acc_policy, 'best': acc_best}


@task(use_cache=True)
async def train_baseline(dataset, lr, fraction, epochs, tau, seed):
    train = await load_train(dataset)
    baseline_size = int(fraction * train.n)
    policy = create_policy(train.d, train.k, 'boltzmann', tau=tau)
    model = policy.create()
    prng = rng_seed(seed)
    indices = prng.permutation(train.n)[0:baseline_size]
    logging.info(f"[{seed:4d}] Training baseline ({lr}, {tau})")
    optimize_supervised_hinge(train, indices, model, lr, epochs)
    return model


if __name__ == "__main__":
    main()
