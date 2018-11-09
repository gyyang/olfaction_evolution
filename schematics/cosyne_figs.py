import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools

mpl.rcParams['font.size'] = 7

new_rootpath = os.path.join(rootpath, 'files', 'vary_size_experiment')
fig_dir = os.path.join(os.getcwd())


def _load_results(save_name):
    dir = os.path.join(new_rootpath, save_name, 'files')
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    
    
    glo_score = np.zeros(len(dirs))
    val_acc = np.zeros(len(dirs))
    n_pns = np.zeros(len(dirs))
    n_kcs = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config = tools.load_config(d)
        glo_score[i] = log['glo_score'][-1]
        val_acc[i] = log['val_acc'][-1]
        n_pns[i] = config.N_GLO
        n_kcs[i] = config.N_KC
    return glo_score, val_acc, n_pns, n_kcs
    
glo_score, val_acc, n_pns, n_kcs = _load_results('vary_PN')
ind_sort = np.argsort(n_pns)

fig = plt.figure(figsize=(2, 1.2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
ax.plot(np.log10(n_pns[ind_sort]), glo_score[ind_sort], 'o-', markersize=3)
ax.set_xlabel('Number of PNs')
ax.set_ylabel('GloScore')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
plt.xticks(np.log10([10, 50, 100, 250]), [10, 50, 100, 250])
plt.savefig(os.path.join(fig_dir, 'vary_PN.pdf'))


glo_score, val_acc, n_pns, n_kcs = _load_results('vary_KC')
ind_sort = np.argsort(n_kcs)

fig = plt.figure(figsize=(2, 1.2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
ax.plot(np.log10(n_kcs[ind_sort]), glo_score[ind_sort], 'o-', markersize=3)
ax.set_xlabel('Number of KCs')
ax.set_ylabel('GloScore')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
plt.xticks(np.log10([50, 200, 2500, 20000]), [50, 200, '2.5K', '20K'])

plt.savefig(os.path.join(fig_dir, 'vary_KC.pdf'))
