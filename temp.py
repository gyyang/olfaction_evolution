"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa
import pickle
import standard.analysis_activity as analysis_activity

import matplotlib.pyplot as plt

def t(experiment, save_path,s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)


def st(experiment, save_path, s=0,e=1000):
    """Train all models locally."""
    for i in range(s, e):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)

# def temp_norm():
#     config = configs.FullConfig()
#     config.data_dir = './datasets/proto/mask'
#     config.max_epoch = 10
#
#     config.direct_glo = True
#     config.initializer_orn2pn = 'constant'
#
#     config.N_ORN_DUPLICATION = 1
#     config.ORN_NOISE_STD = 0
#     # config.train_kc_bias = False
#     # config.train_pn2kc = True
#     # config.sparse_pn2kc = False
#
#     # Ranges of hyperparameters to loop over
#     hp_ranges = OrderedDict()
#     hp_ranges['data_dir'] = ['./datasets/proto/standard', './datasets/proto/concentration']
#     hp_ranges['pn_norm_post'] = ['None', 'biology']
#     return config, hp_ranges

def make_datafiles():
    config = configs.input_ProtoConfig()
    for i in [20, 40, 60, 80, 100, 120, 140, 160, 200, 500, 1000]:
        config.n_trueclass = i
        config.N_CLASS = 20
        config.relabel = True
        task.save_proto(config, str(config.n_trueclass) + '_' + str(config.N_CLASS))
        print('Done: ' + str(i))
# make_datafiles()

#TODO: push this into experiments
def why_kc_layer():
    config = configs.FullConfig()
    config.max_epoch = 8
    config.model = 'full'
    # config.model = 'normmlp'

    config.NEURONS = []
    config.kc_dropout = True
    config.kc_dropout_rate = 0
    config.direct_glo = True
    config.initializer_orn2pn = 'constant'
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    x = [20, 40, 60, 80, 100, 120, 140, 160, 200, 500, 1000]
    datasets = ['./datasets/proto/_s' + str(i) + '_20' for i in x]
    hp_ranges['model'] = ['normmlp', 'full']
    hp_ranges['data_dir'] = datasets
    return config, hp_ranges

# path = './files_temp/relabel_layers'
# t(temp_oracle(), path, s=0, e=100)
# sa.plot_results(path, x_key='n_trueclass', y_key='val_acc', loop_key='model')


def temp():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 15

    config.replicate_orn_with_tiling = True
    config.direct_glo = True
    config.initializer_orn2pn = 'constant'
    config.kc_dropout = True

    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [1000, 2500, 5000, 10000, 20000]
    x = [40, 80, 200, 500, 1000]
    hp_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_20' for i in x]
    return config, hp_ranges

# path = './files_temp/temp_nkc'
# t(temp(), path, s=0, e=100)
# sa.plot_results(path, x_key='n_trueclass', y_key='val_acc', loop_key='N_KC')

def temp_rnn():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 12
    config.model = 'full'

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_norm_post = 'batch_norm'
    # config.NEURONS = 2500
    # config.TIME_STEPS =1
    # config.WEIGHT_LOSS = False
    # config.WEIGHT_ALPHA = 3
    #
    # config.dropout = True
    # config.dropout_rate = .5

    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [True]
    return config, hp_ranges

def plot_weights(root_path, var_name = 'w_orn', sort = True, sort_axis = 0, double_sort = 1, dir_ix = 0):
    """Plot weights.

    Currently this plots OR2ORN, ORN2PN, and OR2PN
    """
    #TODO: fix code
    dirs = [os.path.join(root_path, n) for n in os.listdir(root_path)]
    save_path = dirs[dir_ix]
    config = tools.load_config(save_path)
    # Load network at the end of training
    model_dir = os.path.join(save_path, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_plot = var_dict[var_name]

    # Sort for visualization
    if sort:
        if sort_axis == 0:
            ind_max = np.argmax(w_plot, axis=0)
            ind_sort = np.argsort(ind_max)
            w_plot = w_plot[:, ind_sort]
            if double_sort:
                w_plot = w_plot[ind_sort,:]

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_axes(rect)
    vlim = np.round(np.max(abs(w_plot)), decimals=1)
    im = ax.imshow(w_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    sa._easy_save(root_path, '_' + var_name + '_' + str(dir_ix))

path = './files_temp/RNN'
t(temp_rnn(), path, s=0, e=100)

dir_ix = 0
var_name = 'w_glo'
# plot_weights(path, var_name = var_name, sort = False, double_sort=False, sort_axis= 0, dir_ix = dir_ix)
# plot_weights(path, var_name = 'w_out', sort = False, double_sort=False, sort_axis= 0, dir_ix = 0)

dirs = [os.path.join(path, n) for n in os.listdir(path)]
save_path = dirs[dir_ix]
config = tools.load_config(save_path)
model_dir = os.path.join(save_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_plot = var_dict[var_name]

# ind_max = np.argmax(w_plot, axis=0)
# ind_sort = np.argsort(ind_max)
# w_plot = w_plot[:, ind_sort]
# w_plot = w_plot[ind_sort,:]

pn2kc = w_plot
# pn2kc = np.sort(pn2kc, axis=0)[::-1,:]
# plt.imshow(pn2kc)
# plt.show()
# mean = np.mean(pn2kc, axis=1)
# plt.plot(mean)
# plt.show()

plt.hist(pn2kc.flatten(),bins=50, range=[.03, .5], density=False)
plt.show()

threshold = 0.05
claw_count = np.sum(pn2kc>threshold,axis=0)
plt.hist(claw_count, bins=50, range=[0,50], density=False)
plt.show()