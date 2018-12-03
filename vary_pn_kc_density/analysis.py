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

new_rootpath = os.path.join(rootpath, 'files', 'vary_pn_kc_density')
fig_dir = os.path.join(rootpath, 'figures')


def _load_results():
    dir = new_rootpath
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]

    glo_score = np.zeros(len(dirs))
    val_acc = np.zeros(len(dirs))
    kc_inputs_list = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config = tools.load_config(d)
        glo_score[i] = log['glo_score'][-1]
        val_acc[i] = log['val_acc'][-1]
        kc_inputs_list[i] = config.kc_inputs
    return glo_score, val_acc, kc_inputs_list


glo_score, val_acc, kc_inputs_list = _load_results()
ind_sort = np.argsort(kc_inputs_list)

xticks = [3, 10, 20, 50]

fig = plt.figure(figsize=(2, 1.2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
ax.plot(np.log10(kc_inputs_list[ind_sort]), glo_score[ind_sort], 'o-', markersize=3)
ax.plot(np.log10([7, 7]), [0, 1], '--', color='gray')
ax.set_xlabel('Number of input connections for each KC')
ax.set_ylabel('GloScore')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
plt.xticks(np.log10(xticks), xticks)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'gloscore_vs_kc_inputs.pdf'), transparent=True)


fig = plt.figure(figsize=(2, 1.2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
ax.plot(np.log10(kc_inputs_list[ind_sort]), val_acc[ind_sort], 'o-', markersize=3)
ax.plot(np.log10([7, 7]), [0, 1], '--', color='gray')
ax.set_xlabel('Number of input connections for each KC')
ax.set_ylabel('Validation accuracy')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
plt.xticks(np.log10(xticks), xticks)
# plt.show()
plt.savefig(os.path.join(fig_dir, 'acc_vs_kc_inputs.pdf'), transparent=True)