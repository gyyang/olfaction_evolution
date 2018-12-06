import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils

param = "kc_inputs"
condition = "./vary_KC_claws/uniform_bias_trainable_simple_meansub"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]
titles = ['glo score', 'val acc', 'val loss', 'train loss']

worn = utils.load_pickle(dir, 'w_orn')
born = utils.load_pickle(dir, 'model/layer1/bias:0')
wglo = utils.load_pickle(dir, 'w_glo')
bglo = utils.load_pickle(dir, 'model/layer2/bias:0')

dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
kcs = []
for d in dirs:
    glo_act, glo_in, glo_in_pre, kc = utils.get_model_vars(d)
    kcs.append(kc)

def corr(mat):
    max_n = 2000
    #every row is an observation
    n = mat.shape[0]
    if n > max_n:
        mat = mat[:max_n,:]
        n = max_n
    corr_mat = np.corrcoef(mat)
    mask = ~np.eye(n, dtype=bool)
    val = np.mean(corr_mat[mask])
    return val, corr_mat

#matrix correlation
corr_kc_matrix, _ = zip(*[corr(mat) for mat in wglo])

#activity correlation
corr_kc_activity, _ = zip(*[corr(mat) for mat in kcs])

last_val_acc = [x[-1] for x in val_acc]
nr, nc = 2, 2
fig, ax = plt.subplots(nrows = nr, ncols = nc)
cur_ax = ax[0,0]
cur_ax.scatter(parameters, corr_kc_activity)
cur_ax.set_xlabel(param)
cur_ax.set_ylabel('average kc activity correlation')

cur_ax = ax[1,0]
cur_ax.scatter(corr_kc_matrix, corr_kc_activity)
cur_ax.set_xlabel('average kc weight correlation')
cur_ax.set_ylabel('average kc activity correlation')

cur_ax = ax[1,1]
cur_ax.scatter(corr_kc_activity, last_val_acc)
cur_ax.set_xlabel('average kc activity correlation')
cur_ax.set_ylabel('validation accuracy')

plt.savefig(os.path.join(fig_dir, 'kc_analysis.png'))