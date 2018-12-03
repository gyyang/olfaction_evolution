import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from matplotlib import pylab
import utils
from utils import load_results, adjust

mpl.rcParams['font.size'] = 12
plt.style.use('dark_background')

fig_dir = 'C:/Users/Peter/Dropbox/TALKS/LAB MEETINGS/2018.11.29'
fig_dir = os.path.join(fig_dir)
root_dir = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/duplication'
save_name = '.25_noise'

glo_score, val_acc, config = load_results(root_dir, save_name)
params = np.array([''])

ix = [0]
glo_score = [glo_score[x] for x in ix]
val_acc = [val_acc[x] for x in ix]
legends = params[ix]
number = len(legends)
cmap = plt.get_cmap('cool')
colors = [cmap(i) for i in np.linspace(0, 1, number)]
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 5))

cur_ax = ax[1]
plt.sca(cur_ax)
adjust(cur_ax)
cur_ax.set_color_cycle(colors)
cur_ax.plot(np.array(glo_score).transpose())
cur_ax.legend(legends, loc='lower right', fancybox=True, framealpha=0)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('GloScore')

cur_ax = ax[0]
adjust(cur_ax)
cur_ax.set_color_cycle(colors)
cur_ax.plot(np.array(val_acc).transpose())
cur_ax.legend(legends, loc='lower right', fancybox=True, framealpha=0)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Validation Accuracy')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'one_example_summary.png'), transparent=True)



# glo_score, val_acc, n_pns, n_kcs = _load_results('vary_KC')
# ind_sort = np.argsort(n_kcs)
#
# fig = plt.figure(figsize=(2, 1.2))
# ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
# ax.plot(np.log10(n_kcs[ind_sort]), glo_score[ind_sort], 'o-', markersize=3)
# ax.plot(np.log10([2500, 2500]), [0, 1], '--', color='gray')
# ax.set_xlabel('Number of KCs')
# ax.set_ylabel('GloScore')
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# ax.yaxis.set_ticks([0, 0.5, 1.0])
# ax.xaxis.set_ticks([])
# plt.xticks(np.log10([50, 200, 2500, 20000]), [50, 200, '2.5K', '20K'])
#
# plt.savefig(os.path.join(fig_dir, 'vary_KC.png'), transparent=True)
