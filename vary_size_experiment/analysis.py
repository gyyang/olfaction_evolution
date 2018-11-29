import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils

param = "N_PN"
condition = "vary_PN"

mpl.rcParams['font.size'] = 10
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]
titles = ['glo score', 'val acc', 'val loss', 'train loss']
# utils.plot_summary(data, titles, fig_dir, list_of_legends, param)
# utils.plot_summary_last_epoch(data, titles, fig_dir, parameters, param)

worn = utils.load_pickle(dir, 'w_orn')
born = utils.load_pickle(dir, 'model/layer1/bias:0')
wpn = utils.load_pickle(dir, 'model/layer2/kernel:0')
bpn = utils.load_pickle(dir, 'model/layer2/bias:0')

# for p, cur_w in zip(parameters, worn):
    # utils.plot_weights(cur_w, str(p), arg_sort = 1, fig_dir = fig_dir)


# # Reload the network and analyze activity
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
fig, ax = plt.subplots(nrows=4, ncols=2)
for i, d in enumerate(dirs):
    glo_act, glo_in, glo_in_pre, kc = utils.get_model_vars(d)

    ax[i,0].hist(kc.flatten(), bins=100, range =(0, 5))
    ax[i,0].set_title('Activity: ' + str(list_of_legends[i]))
    sparsity = np.count_nonzero(kc >0, axis= 1) / kc.shape[1]
    ax[i,1].hist(sparsity, bins=20, range=(0,1))
    ax[i,1].set_title('Sparseness')

plt.savefig(os.path.join(fig_dir, 'activity.png'))