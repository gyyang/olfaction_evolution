import matplotlib.pyplot as plt
import pickle
import numpy as np
import tools
from pylab import *
import matplotlib.animation as animation
import matplotlib as mpl
import numpy as np
import os
import glob
import standard.analysis as sa
import tools
import matplotlib.pyplot as plt
import task
import tensorflow as tf
from model import FullModel
from dict_methods import *

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def _get_K(res):
    n_model, n_epoch = res['sparsity'].shape[:2]
    Ks = np.zeros((n_model, n_epoch))
    bad_KC = np.zeros((n_model, n_epoch))
    for i in range(n_model):
        for j in range(n_epoch):
            sparsity = res['sparsity'][i, j]
            Ks[i, j] = sparsity[sparsity>0].mean()
            bad_KC[i,j] = np.sum(sparsity==0)/sparsity.size
    res['K'] = Ks
    res['bad_KC'] = bad_KC


def simple_plot(xkey, ykey, filter_dict=None):
    if filter_dict is not None:
        temp = filter(res, filter_dict=filter_dict)

    x = temp[xkey]
    y = temp[ykey][:, -1]
    plt.figure()
    plt.plot(np.log(x), y, '*')
    plt.xticks(np.log(x), x)
    plt.xlabel(xkey)
    plt.ylabel(ykey)

    # if filter_dict is not None:
    #   plt.legend('{} = {}'.format(filter_dict.key[0],filter_dict.value[0]))

def marginal_plot(xkey, ykey, vary_key, select_dict=None):
    plt.figure(figsize=(3,2))
    for i in np.unique(res[vary_key]):
        temp = filter(res, {vary_key:i})
        if select_dict:
            temp = filter(temp, select_dict)
        x = temp[xkey]
        y = temp[ykey][:,-1]

        plt.plot(np.log(x), y, '*')
    x = np.unique(res[xkey])
    plt.xticks(np.log(x),x)
    plt.xlabel(xkey)
    plt.ylabel(ykey)
    plt.legend(np.unique(res[vary_key]))
    plt.title(vary_key)

    figname = '_' + ykey + '_vs_' + xkey
    if vary_key:
        figname += '_vary' + vary_key
    if select_dict:
        for k, v in select_dict.items():
            if isinstance(v, list):
                v = [x.rsplit('/', 1)[-1] for x in v]
                v = str('__'.join(v))
            else:
                v = str(v)
            figname += k + '_' + v + '__'

    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    sa._easy_save(d, figname)

# d = os.path.join(os.getcwd(), 'files', 'cluster_simple', 'cluster_simple50')
d = os.path.join(os.getcwd(), 'files_temp', 'cluster_big50')
files = glob.glob(d)
print(len(files))
res = defaultdict(list)
for f in files:
    temp = tools.load_all_results(f, argLast = False)
    chain_defaultdicts(res, temp)


peak_inds = np.zeros_like(res['kc_prune_threshold']).astype(np.bool)
for i, thres in enumerate(res['kc_prune_threshold']):
    x = np.where(res['lin_bins'][0,:-1] > res['kc_prune_threshold'][i])[0][0]
    peak = np.argmax(res['lin_hist'][i, -1, (x-10):])
    if thres == 0.08 and res['N_KC'][i] == 2500:
        print('wtf')
    if peak > 20:
        peak_inds[i] = True
    else:
        peak_inds[i] = False

acc_ind = res['train_acc'][:,-1] > .4
badkc_ind = res['bad_KC'][:,-1] < .2
ind = badkc_ind * acc_ind * peak_inds

for k, v in res.items():
    res[k] = v[ind]

_get_K(res)
marginal_plot('lr', 'K', 'kc_prune_threshold', {'N_KC':2500, 'kc_dropout_rate':0.6})
marginal_plot('lr', 'K', 'N_KC', {'kc_prune_threshold':0.1, 'kc_dropout_rate':0.6})
marginal_plot('lr', 'K', 'kc_dropout_rate', {'kc_prune_threshold':0.1, 'N_KC':2500})

# plt.figure(figsize=(3,2))
# x = filter(res, {'kc_prune_threshold':0.08, 'N_KC':2500})
# plt.plot(x['lin_bins'][0,:-1],x['lin_hist'][:,-1].T)
# plt.ylim([0, 300])
# plt.legend(x['lr'])
# ax = plt.gca()
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# sa._easy_save(d, 'pn2kc_weight_distribution')
#
# plt.figure(figsize=(3,2))
# x = filter(x, {'N_KC':2500, 'kc_prune_threshold':0.08})
# plt.plot(x['val_logloss'].T)
# plt.legend(np.unique(x['separate_lr']))
# plt.legend(x['lr'])
# ax = plt.gca()
# ax.spines["right"].set_visible(False)
# ax.spines["top"].set_visible(False)
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
# sa._easy_save(d, 'training_speed')