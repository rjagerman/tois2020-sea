import logging
import json
from argparse import ArgumentParser
from argparse import Namespace


import numpy as np
import scipy.stats as st
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = '\\usepackage{biolinum}\n\\usepackage{sfmath}\n\\usepackage[T1]{fontenc}\n\\usepackage[libertine]{newtxmath}' #\\usepackage{libertine}\n
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams.update({'font.size': 16})
from matplotlib import pyplot as plt
from experiments.util import rng_seed, get_evaluation_points, mkdir_if_not_exists, NumpyEncoder


def to_np_arrays(structure):
    if isinstance(structure, list):
        out = np.array(structure)
        if out.dtype == np.object:
            return [to_np_arrays(s) for s in structure]
        else:
            return out
    if isinstance(structure, dict):
        return {
            k: to_np_arrays(v) for k, v in structure.items()
        }
    return structure


plot_height = 2.0

methods = {
    'baseline': '$\\pi_b$',
    'epsgreedy': '$\\epsilon$-greedy',
    'boltzmann': 'Boltzmann',
    'ips': 'IPS',
    'sea': 'SEA',
    'ucb': 'UCB',
    'thompson': 'Thompson',
    'online': 'RankSVM (Online)',
    'duelingbandit': 'DBGD',
    'meancomp': 'BSEA'
}

behaviors = {
    'perfect': 'Perfect clicks',
    'position': 'Position-biased clicks',
    'nearrandom': 'Near random clicks'
}
cbehaviors = {
    'perfect': 'Perfect rewards',
    'noise': 'Noisy rewards (0.9 / 0.1)',
    'extreme': 'Near random rewards (0.6 / 0.4)'
}

markers = {
    'baseline': 'o',
    'boltzmann': 's',
    'epsgreedy': 'd',
    'online': 'd',
    'duelingbandit': 's',
    'ips': 'v',
    'sea': '*',
    'meancomp': 'H',
    'ucb': '<',
    'thompson': '>'
}

colors = {
    'baseline': 'grey',
    'boltzmann': 'C2',
    'epsgreedy': 'C1',
    'online': 'C8',
    'ips': 'C3',
    'sea': 'C0',
    'meancomp': 'C6',
    'duelingbandit': 'C11',
    'ucb': 'C4',
    'thompson': 'C5'
}

sorting = {
    'sea': 0,
    'meancomp': 0.5,
    'ips': 7,
    'boltzmann': 3,
    'online': 3.5,
    'epsgreedy': 4,
    'duelingbandit': 4.5,
    'ucb': 5,
    'thompson': 6,
    'baseline': 1
}

datasets = {
    'mslr10k': 'MSLR10k',
    'istella-s': 'Istella-s',
    'yahoo': 'Webscope',
    'usps': 'USPS',
    'news20': '20-News',
    'rcv1': 'RCV1'
}


def plot_classification():
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(11, plot_height*4.5)) #*1.5
    plot_count = 0
    for dataset in ['usps', 'news20', 'rcv1']:
        for behavior in ["perfect", "extreme"]:
            plot_count += 1
            ax = plt.subplot(320 + plot_count)
            with open(f'./results/classification/{dataset}/{behavior}.json', 'rt') as f:
                data = json.load(f)
                data = to_np_arrays(data)
            for method in sorted(data, key=lambda e: sorting[e['args']['strategy']], reverse=True):
                strategy = method['args']['strategy']
                if strategy == 'greedy':
                    continue
                if strategy == 'ips':
                    strategy = 'baseline'
                r = method['result']
                x = r['x']
                y = r['deploy']['mean']
                ystd = r['deploy']['std']
                plt.plot(x, y, label=methods[strategy], color=colors[strategy], markevery=0.1, marker=markers[strategy])
                plt.fill_between(x, y - ystd, y + ystd, alpha=0.35, color=colors[strategy])
                plt.xscale('symlog')
            if plot_count <= 2:
                plt.title(f"{cbehaviors[behavior]}")
            if behavior == 'perfect':
                plt.ylabel('Reward $r$')
            if plot_count > 4:
                plt.xlabel('Round $t$')
            if plot_count == 1:
                handles, labels = plt.gca().get_legend_handles_labels()
            if plot_count == 2:
                lgd = plt.legend(reversed(handles), reversed(labels), ncol=4, bbox_to_anchor=(-0.1, 1.4), loc='center')
    plt.savefig(f'./plots/classification.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')


def print_classification_regret_table():
    def sortmethods(a):
        return {
            'ucb': 1,
            'thompson': 2,
            'epsgreedy': 3,
            'boltzmann': 4,
            'sea': 5,
            'ips': 6,
            'greedy': 7,
            'meancomp': 4.5
        }[a['args']['strategy']]

    tblx = [100, 1000, 10000, 100000, 1000000]
    exps = [str(i) if i <= 1 else f'$10^{{{int(np.log10(i))}}}$' for i in tblx]

    print(f"\\begin{{tabular}}{{ll@{{\\hspace{{1cm}}}}r{'r'.join(['' for _ in exps])}}}")
    print(f"  \\toprule")
    print(f"    & \\bf {'Round $t$ ':17s} & {' & '.join(exps)} \\\\")
    for behavior in ["perfect", "extreme"]:
        print("  \\midrule")
        print(f"    \\multicolumn{{7}}{{c}}{{\\emph{{{cbehaviors[behavior]}}}}}\\\\")
        print(f"  \\midrule")
        for dataset in ['usps', 'news20', 'rcv1']:
            with open(f'./results/classification/{dataset}/{behavior}.json', 'rt') as f:
                data = json.load(f)
                data = to_np_arrays(data)
            count_rows = 0
            for method in sorted(data, key=sortmethods):
                if method['args']['strategy'] in ['greedy', 'ips']:
                    continue
                count_rows += 1
            print(f"    \\multirow{{{count_rows}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{datasets[dataset]}}}}}")
            baseline = [d['result']['regret'] for d in data if d['args']['strategy'] == 'ips'][0]
            sea_base = [d['result']['regret'] for d in data if d['args']['strategy'] == 'sea'][0]
            for method in sorted(data, key=sortmethods):
                if method['args']['strategy'] in ['greedy', 'ips']:
                    continue
                if method['args']['strategy'] in ['meancomp'] or behavior == 'extreme':
                    prefix, suffix = "\\added{", "}"
                else:
                    prefix, suffix = "", ""
                r = method['result']
                print(f"      & {prefix}{methods[method['args']['strategy']]:17s}{suffix}", end='')
                for i in range(len(r['x'])):
                    if r['x'][i] in tblx:
                        m1 = sea_base['mean'][i]
                        m2 = r['regret']['mean'][i]
                        s1 = sea_base['std'][i]
                        s2 = r['regret']['std'][i]
                        n1 = n2 = 5
                        tt = st.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
                        sigfc = ''
                        if tt.pvalue < 0.01:
                            if tt.statistic >= 0:
                                sigfc = '\\rlap{$^\\blacktriangle$}'
                            else:
                                sigfc = '\\rlap{$^\\blacktriangledown$}'
                        print(f" & {prefix}{1.0 * (baseline['mean'][i] - r['regret']['mean'][i]):.2f}{sigfc}{suffix}", end=' ')

                if method['args']['strategy'] == "sea" and dataset != "rcv1":
                    print('\\\\[2ex]')
                    print("%")
                else:
                    print('\\\\')

    print(f"  \\bottomrule")
    print(f"\\end{{tabular}}")


def print_classification_perf_table():
    def sortmethods(a):
        return {
            'ucb': 1,
            'thompson': 2,
            'epsgreedy': 3,
            'boltzmann': 4,
            'sea': 7,
            'ips': 6,
            'greedy': 5,
            'meancomp': 6.5,
            'online': 1,
            'duelingbandit': 1.5
        }[a['args']['strategy']]

    print(f"\\begin{{tabular}}{{ll@{{\\hspace{{1cm}}}}rr}}")
    print(f"  \\toprule")
    print("     & ")
    for behavior in ['perfect', 'extreme']:
        print(f" & \\emph{{{cbehaviors[behavior]:17s}}}", end='')
    print(f"\\\\")
    print(f"  \\midrule")
    for dataset in ['usps', 'news20', 'rcv1']:
        if dataset == 'usps':
            rowcount = 7
        else:
            rowcount = 5
        print(f"    \\multirow{{{rowcount}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{datasets[dataset]}}}}}")
        for method in ['epsgreedy', 'boltzmann', 'ucb', 'thompson', 'ips', 'meancomp', 'sea']:
            if method in ['ucb', 'thompson'] and dataset != 'usps':
                continue
            if method == 'meancomp':
                prefix, suffix = "\\added{", "}"
            else:
                prefix, suffix = "", ""
            print(f"      & {prefix}{methods[method]}{suffix}", end='')
            for behavior in ['perfect', 'extreme']:
                if method == 'meancomp' or behavior == 'extreme':
                    prefix, suffix = "\\added{", "}"
                else:
                    prefix, suffix = "", ""
                with open(f'./results/classification/{dataset}/{behavior}.json', 'rt') as f:
                    data = json.load(f)
                    data = to_np_arrays(data)
                for m in data:
                    if m['args']['strategy'] == 'sea':
                        baseline = m['result']['learned']
                    if m['args']['strategy'] == method:
                        r = m['result']
                if r is None:
                    print(f" & --", end=' ')
                for i in range(len(r['x'])):
                    if r['x'][i] == 1_000_000:
                        m1 = baseline['mean'][i]
                        m2 = r['learned']['mean'][i]
                        s1 = baseline['std'][i]
                        s2 = r['learned']['std'][i]
                        n1 = n2 = 5
                        tt = st.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
                        sigfc = ''
                        if tt.pvalue < 0.01:
                            if tt.statistic <= 0:
                                sigfc = '\\rlap{$^\\blacktriangle$}'
                            else:
                                sigfc = '\\rlap{$^\\blacktriangledown$}'
                        print(f" & {prefix}{1.0 * (r['learned']['mean'][i]):.2f}{sigfc}{suffix}", end=' ')
            if method == 'sea' and dataset != 'rcv1':
                print(f"\\\\[2ex]")
                print("%")
            else:
                print(f"\\\\")
    print(f"  \\bottomrule")
    print(f"\\end{{tabular}}")


def plot_ranking():
    matplotlib.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(16, plot_height*4.5)) #*1.5
    plot_count = 0
    for dataset in ['mslr10k', 'yahoo', 'istella-s']:
        for behavior in ['perfect', 'position', 'nearrandom']:
            plot_count += 1
            ax = plt.subplot(330 + plot_count)
            with open(f'./results/ranking/{dataset}/{behavior}.json', 'rt') as f:
                data = json.load(f)
                data = to_np_arrays(data)
            for method in sorted(data, key=lambda e: sorting[e['args']['strategy']], reverse=True):
                strategy = method['args']['strategy']
                if strategy == 'greedy':
                    continue
                if strategy == 'ips':
                    strategy = 'baseline'
                r = method['result']
                x = r['x']
                y = r['deploy']['mean']
                ystd = r['deploy']['std']
                plt.plot(x, y, label=methods[strategy], color=colors[strategy], markevery=0.1, marker=markers[strategy])
                plt.fill_between(x, y - ystd, y + ystd, alpha=0.35, color=colors[strategy])
                plt.xscale('symlog')
            if plot_count <= 3:
                plt.title(f"{behaviors[behavior]}")
            if behavior == 'perfect':
                plt.ylabel('NDCG@10')
            if plot_count > 6:
                plt.xlabel('Round $t$')
            if plot_count == 1:
                handles, labels = plt.gca().get_legend_handles_labels()
            if plot_count == 2:
                lgd = plt.legend(reversed(handles), reversed(labels), ncol=3, bbox_to_anchor=(0.5, 1.4), loc='center')
            if dataset == 'mslr10k':
                plt.ylim([0.29, 0.45])
            elif dataset == 'yahoo':
                plt.ylim([0.655, 0.76])
            elif dataset == 'istella-s':
                plt.ylim([0.45, 0.70])
    plt.savefig(f'./plots/ranking.pdf', bbox_extra_artists=[lgd], bbox_inches='tight')


def print_ranking_regret_table():
    def sortmethods(a):
        return {
            'online': 1,
            'ucb': 1,
            'thompson': 2,
            'epsgreedy': 3,
            'boltzmann': 4,
            'sea': 5,
            'ips': 6,
            'greedy': 7,
            'meancomp': 4.5,
            'duelingbandit': 1.5
        }[a['args']['strategy']]

    tblx = [100, 1000, 10000, 100000, 1000000, 10000000]
    exps = [str(i) if i <= 1 else f'$10^{{{int(np.log10(i))}}}$' for i in tblx]


    print(f"\\begin{{tabular}}{{ll@{{\\hspace{{1cm}}}}r{'r'.join(['' for _ in exps])}}}")
    print(f"  \\toprule")
    print(f"    & \\bf {'Round $t$ ':17s} & {' & '.join(exps)} \\\\")
    for behavior in ['perfect', 'position', 'nearrandom']:
        print("  \\midrule")
        print(f"    \\multicolumn{{8}}{{c}}{{\\emph{{{behaviors[behavior]}}}}}\\\\")
        print(f"  \\midrule")
        for dataset in ['mslr10k', 'yahoo', 'istella-s']:
            with open(f'./results/ranking/{dataset}/{behavior}.json', 'rt') as f:
                data = json.load(f)
                data = to_np_arrays(data)
            print(f"    \\multirow{{4}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{datasets[dataset]}}}}}")
            baseline = [d['result']['regret'] for d in data if d['args']['strategy'] == 'ips'][0]
            for method in sorted(data, key=sortmethods):
                if method['args']['strategy'] in ['greedy', 'ips']:
                    continue
                if method['args']['strategy'] in ['meancomp', 'duelingbandit']:
                    prefix, suffix = "\\added{", "}"
                else:
                    prefix, suffix = "", ""
                r = method['result']
                print(f"     & {prefix}{methods[method['args']['strategy']]:17s}{suffix}", end='')
                for i in range(len(r['x'])):
                    if r['x'][i] in tblx:
                        m1 = baseline['mean'][i]
                        m2 = r['regret']['mean'][i]
                        s1 = baseline['std'][i]
                        s2 = r['regret']['std'][i]
                        n1 = n2 = 5
                        tt = st.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
                        sigfc = ''
                        if tt.pvalue < 0.01:
                            if tt.statistic >= 0:
                                sigfc = '\\rlap{$^\\blacktriangle$}'
                            else:
                                sigfc = '\\rlap{$^\\blacktriangledown$}'
                        print(f" & {prefix}{1.0 * (baseline['mean'][i] - r['regret']['mean'][i]):.2f}{sigfc}{suffix}", end=' ')
                if method['args']['strategy'] == 'sea' and dataset != 'istella-s':
                    print('\\\\[2ex]')
                    print("%")
                else:
                    print('\\\\')
    print(f"\\bottomrule")
    print(f"\\end{{tabular}}")


def print_ranking_perf_table():
    def sortmethods(a):
        return {
            'ucb': 1,
            'thompson': 2,
            'epsgreedy': 3,
            'boltzmann': 4,
            'sea': 7,
            'ips': 6,
            'greedy': 5,
            'meancomp': 6.5,
            'online': 1,
            'duelingbandit': 1.5
        }[a['args']['strategy']]

    print(f"\\begin{{tabular}}{{ll@{{\\hspace{{1cm}}}}rrr}}")
    print(f"  \\toprule")
    print(f"    & ")
    for behavior in ['perfect', 'position', 'nearrandom']:
        print(f" & \\emph{{{behaviors[behavior]:17s}}}", end='')
    print(f"\\\\")
    print(f"  \\midrule")
    for dataset in ['mslr10k', 'yahoo', 'istella-s']:
        print(f"    \\multirow{{5}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{datasets[dataset]}}}}}")
        for method in ['online', 'duelingbandit', 'ips', 'meancomp', 'sea']:
            if method in ['duelingbandit', 'meancomp']:
                prefix, suffix = "\\added{", "}"
            else:
                prefix, suffix = "", ""
            print(f"      & {prefix}{methods[method]}{suffix}", end='')
            for behavior in ['perfect', 'position', 'nearrandom']:
                with open(f'./results/ranking/{dataset}/{behavior}.json', 'rt') as f:
                    data = json.load(f)
                    data = to_np_arrays(data)
                for m in data:
                    if m['args']['strategy'] == 'sea':
                        baseline = m['result']['learned']
                    if m['args']['strategy'] == method:
                        r = m['result']
                if r is None:
                    print(f" & --", end=' ')
                for i in range(len(r['x'])):
                    if r['x'][i] == 10_000_000:
                        m1 = baseline['mean'][i]
                        m2 = r['learned']['mean'][i]
                        s1 = baseline['std'][i]
                        s2 = r['learned']['std'][i]
                        n1 = n2 = 5
                        tt = st.ttest_ind_from_stats(m1, s1, n1, m2, s2, n2)
                        sigfc = ''
                        if tt.pvalue < 0.01:
                            if tt.statistic <= 0:
                                sigfc = '\\rlap{$^\\blacktriangle$}'
                            else:
                                sigfc = '\\rlap{$^\\blacktriangledown$}'
                        print(f" & {prefix}{1.0 * (r['learned']['mean'][i]):.2f}{sigfc}{suffix}", end=' ')
            if method == 'sea' and dataset != 'istella-s':
                print(f"\\\\[2ex]")
                print("%")
            else:
                print(f"\\\\")
    print(f"  \\bottomrule")
    print(f"\\end{{tabular}}")


def main():
    # Classification
    plot_classification()
    print("")
    print("============ CLASSIFICATION REGRET TABLE =================")
    print_classification_regret_table()
    print("")
    print("============= CLASSIFICATION PERF TABLE ==================")
    print_classification_perf_table()

    # Ranking
    plot_ranking()
    print("")
    print("=============== RANKING REGRET TABLE =====================")
    print_ranking_regret_table()
    print("")
    print("================ RANKING PERF TABLE ======================")
    print_ranking_perf_table()


if __name__ == "__main__":
    main()
