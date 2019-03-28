from ltrpy.dataset import load
from rulpy.pipeline import task
import logging
import json
import os


with open("conf/ranking/datasets.json", "rt") as f:
    datasets = json.load(f)


@task(use_cache=True)
async def load_from_path(path, filter_queries=False):
    logging.info(f"Loading ranking dataset from {path}")
    return load(path, filter_queries=filter_queries, normalize=True)


@task
async def load_train(dataset, seed=0):
    info = datasets[dataset]
    path_prefix = f"Fold{1 + (seed % info['folds'])}" if info['has_folds'] else ""
    file_path = os.path.join(info['path'], path_prefix, 'train.txt')
    return await load_from_path(file_path, filter_queries=True)


@task
async def load_test(dataset, seed=0):
    info = datasets[dataset]
    path_prefix = f"Fold{1 + (seed % info['folds'])}" if info['has_folds'] else ""
    file_path = os.path.join(info['path'], path_prefix, 'test.txt')
    return await load_from_path(file_path, filter_queries=True)
