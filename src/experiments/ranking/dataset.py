from ltrpy.dataset import load_svmrank as load
from rulpy.pipeline import task
import logging
import json
import os


with open("conf/ranking/datasets.json", "rt") as f:
    datasets = json.load(f)


@task(use_cache=True)
async def load_from_path(path, filter_queries=False, sample=None, sample_inverse=False, seed=42):
    logging.info(f"Loading ranking dataset from {path} (sample:{sample} inv:{sample_inverse}, seed:{seed})")
    dataset = load(path, filter_queries=filter_queries, normalize=True,
                   subsample=sample, subsample_inverse=sample_inverse,
                   subsample_seed=seed)
    return dataset


@task
async def load_train(dataset, seed=0, vali=None):
    if vali is None:
        seed = 0
    info = datasets[dataset]
    path_prefix = f"Fold{1 + (seed % info['folds'])}" if info['has_folds'] else ""
    file_path = os.path.join(info['path'], path_prefix, 'train.txt')
    return await load_from_path(file_path, filter_queries=False, sample=vali, sample_inverse=True, seed=seed)


@task
async def load_test(dataset, seed=0, vali=None):
    info = datasets[dataset]
    path_prefix = f"Fold{1 + (seed % info['folds'])}" if info['has_folds'] else ""
    if vali is not None:
        if info['has_vali']:
            file_path = os.path.join(info['path'], path_prefix, 'vali.txt')
        else:
            file_path = os.path.join(info['path'], path_prefix, 'train.txt')
            return await load_from_path(file_path, filter_queries=False,
                                        sample=vali, sample_inverse=False,
                                        seed=seed)
    else:
        file_path = os.path.join(info['path'], path_prefix, 'test.txt')
    return await load_from_path(file_path, filter_queries=True, seed=0)
