import logging
import json
from argparse import ArgumentParser
from joblib.memory import Memory
from rulpy.pipeline import task, TaskExecutor
from experiments.classification.train import run_experiment, ExperimentConfig, train_baseline


logging.basicConfig(format="[%(asctime)s] %(levelname)s %(threadName)s: %(message)s",
                    level=logging.INFO)


def main():

    cli_parser = ArgumentParser()
    cli_parser.add_argument("-p", "--parallel", type=int, default=1)
    cli_parser.add_argument("-c", "--config", type=str, required=True)
    args = cli_parser.parse_args()

    parser = ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--iterations", type=int, default=100000)
    parser.add_argument("--evaluations", type=int, default=50)
    parser.add_argument("--eval_scale", choices=('lin', 'log'))
    parser.add_argument("--strategy", type=str, default='epsgreedy')
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.05)
    parser.add_argument("--baseline_sample", type=float, default=0.01)
    parser.add_argument("--baseline_lr", type=float, default=0.01)
    parser.add_argument("--baseline_epochs", type=int, default=50)

    with open(args.config, 'rt') as f:
        lines = f.readlines()
        configs = [parser.parse_args(line.strip().split(" ")) for line in lines]
    
    #train = "/Users/rolfjagerman/Datasets/LibSVM/usps.bz2"
    #test = "/Users/rolfjagerman/Datasets/LibSVM/usps.t.bz2"
    #train = "/Users/rolfjagerman/Datasets/LibSVM/news20.scale.bz2"
    #test = "/Users/rolfjagerman/Datasets/LibSVM/news20.t.scale.bz2"
    #seed = 42
    #scale = 'lin'
    # experiments = [
    #     #f"--train_path {train} --test_path {test} --seed {seed} --strategy ucb --eval_scale {scale}",
    #     #f"--train_path {train} --test_path {test} --seed {seed} --strategy thompson --eval_scale {scale}"
    #     #f"--train_path {train} --test_path {test} --seed {seed} --strategy ips --lr 0.01 --eval_scale {scale}",
    #     #f"--train_path {train} --test_path {test} --seed {seed} --strategy boltzmann --lr 0.1 --eval_scale {scale}",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1e1 --eval_scale {scale} --evaluations 5",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1.0 --eval_scale {scale} --evaluations 5",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1e-1 --eval_scale {scale} --evaluations 5",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1e-2 --eval_scale {scale} --evaluations 5",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1e-4 --eval_scale {scale} --evaluations 5",
    #     f"--train_path {train} --test_path {test} --seed {seed} --strategy epsgreedy --lr 1e-6 --eval_scale {scale} --evaluations 5"
    # ]

    with TaskExecutor(max_workers=args.parallel, memory=Memory("cache", compress=6)):
        results = [run_experiment(config) for config in configs]
    
    for result, config in zip(results, configs):
        logging.info(f"{config.strategy} ({config.lr}) = {result.result['policy'][-1]} ({result.result['best'][-1]})")


if __name__ == "__main__":
    main()
