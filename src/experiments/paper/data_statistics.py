from rulpy.pipeline import task, TaskExecutor
from experiments.ranking.dataset import load_train
from joblib.memory import Memory
from collections import Counter
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{biolinum}\n\\usepackage{sfmath}\n\\usepackage[T1]{fontenc}\n\\usepackage[libertine]{newtxmath}' #\\usepackage{libertine}\n
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import pyplot as plt
import json
import numpy as np


def compute_rel_hist(dataset):
    ys_hist = Counter()
    for i in range(dataset.size):
        ys = dataset[i][1]
        for y in ys:
            ys_hist[y] += 1
    a = np.zeros(len(ys_hist))
    for key in ys_hist.keys():
        a[key] = ys_hist[key]
    return a / np.sum(a)


def compute_avg_docs_per_query(dataset):
    sums = []
    for i in range(dataset.size):
        sums.append(dataset[i][0].shape[0])
    return np.mean(sums)


def print_dataset_summary(dataset, name):
    print(f"==== {name} Dataset ===")
    print(f"Avg docs per query: {compute_avg_docs_per_query(dataset)}")
    hist = compute_rel_hist(dataset)



def main():
    with TaskExecutor(max_workers=2, memory=Memory("cache", compress=6)):
        yahoo = load_train("yahoo")
        mslr10k = load_train("mslr10k")
        istella = load_train("istella-s")
    yahoo = yahoo.result
    mslr10k = mslr10k.result
    istella = istella.result

    print_dataset_summary(yahoo, "Yahoo")
    print_dataset_summary(mslr10k, "MSLR10k")
    print_dataset_summary(istella, "Istella-s")

    datasets = {"yahoo": yahoo, "mslr10k": mslr10k, "istella-s": istella}
    names = {"yahoo": "Yahoo", "mslr10k": "MSLR10k", "istella-s": "Istella-s"}

    print(f"\\begin{{tabular}}{{l@{{\\hspace{{1cm}}}}rrrrr}}")
    print(f"  \\toprule")
    print(f"    & 0 & 1 & 2 & 3 & 4 \\")
    print(f"  \\midrule")
    for key in names.keys():
        name = names[key]
        dataset = datasets[key]
        hist = compute_rel_hist(dataset)
        hist_as_table = " & ".join([f"{n:.2f}" for n in hist])
        print(f"    {name} & {hist_as_table}")
    print(f"  \\bottomrule")
    print(f"\\end{{tabular}}")


if __name__ == "__main__":
    main()
