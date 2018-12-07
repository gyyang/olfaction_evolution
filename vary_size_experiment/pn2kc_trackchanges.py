import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils

param = "uniform_pn2kc"
condition = "test/nodup_loss"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]

nr, nc = 3, 2
fig, ax = plt.subplots(nrows=nr, ncols = nc)
for i, d in enumerate(dirs):
    wglo = utils.load_pickle(os.path.join(d,'epoch'), 'w_glo')
    mask = wglo[0] > 0
    data = wglo[0][mask].flatten()
    ax[i,0].hist(data, bins=100, range=(0, .5))
    changes = (wglo[-1][mask] - wglo[0][mask]).flatten()
    ax[i,1].hist(changes, bins=100, range=(-1.5, 1.5))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'kc_weight_changes.png'))

#
# configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
# parameters = [getattr(config, param) for config in configs]
# list_of_legends = [param +': ' + str(n) for n in parameters]
# data = [glo_score, val_acc, val_loss, train_loss]
# titles = ['glo score', 'val acc', 'val loss', 'train loss']
