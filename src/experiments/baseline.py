import logging
import json
import numpy as np
from argparse import ArgumentParser
from joblib.memory import Memory
from rulpy.pipeline import task, TaskExecutor
from experiments.util import rng_seed
from experiments.classification.train import evaluate_baseline
from scipy import stats as st


logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)s: %(message)s",
                    level=logging.INFO)


def main():

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("-c", "--conf", type=str, required=True)
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--fraction", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=15)
    with open(args.conf, "rt") as f:
        configs = [parser.parse_args(line.strip().split(" ")) for line in f.readlines()]

    with TaskExecutor(max_workers=args.parallel, memory=Memory("cache", compress=6)):
        results = [
            {
                'tau': conf.tau,
                'lr': conf.lr,
                'performance': [
                    evaluate_baseline(conf.train_path, conf.test_path, conf.lr, conf.fraction, conf.epochs, conf.tau, seed)
                    for seed in range(4200, 4200 + conf.repeats)
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


if __name__ == "__main__":
    main()
