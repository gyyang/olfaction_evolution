import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools
from tools import nicename

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')


def plot_results(path, x_key, y_key, loop_key=None):
    res = tools.load_all_results(path)

    # Sort by x_key
    ind_sort = np.argsort(res[x_key])
    for key, val in res.items():
        res[key] = val[ind_sort]

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    if loop_key:
        for x in np.unique(res[loop_key]):
            ind = res[loop_key] == x
            x_plot = res[x_key][ind]
            if x_key == 'N_KC':
                x_plot = np.log(x_plot)
            ax.plot(x_plot, res[y_key][ind], 'o-', label=str(x))
    else:
        ax.plot(res[x_key], res[y_key], 'o-')

    if x_key == 'N_KC':
        xticks = np.array([30, 100, 1000, 10000])
        ax.set_xticks(np.log(xticks))
    else:
        xticks = res[x_key]
        ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlabel(nicename(x_key))
    ax.set_ylabel(nicename(y_key))
    ax.set_yticks([0, 0.5, 1.0])
    plt.ylim([0, 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if loop_key and y_key == 'glo_score':
        l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5))
        l.set_title(nicename(loop_key))

    figname = y_key + 'vs' + x_key
    if loop_key:
        figname += '_vary' + loop_key
    figname = os.path.join(figpath, figname)
    plt.savefig(figname+'.pdf', transparent=True)
    plt.savefig(figname+'.png', dpi=300)


path = os.path.join(rootpath, 'files', 'vary_noise3')
plot_results(path, x_key='N_KC', y_key='glo_score', loop_key='ORN_NOISE_STD')
plot_results(path, x_key='N_KC', y_key='val_acc', loop_key='ORN_NOISE_STD')

# path = os.path.join(rootpath, 'files', 'vary_n_orn_duplication')
# plot_results(path, x_key='N_ORN_DUPLICATION', y_key='glo_score')
# plot_results(path, x_key='N_ORN_DUPLICATION', y_key='val_acc')