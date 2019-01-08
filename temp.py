"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

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

def receptor():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 8

    config.receptor_layer = True
    config.sign_constraint_or2orn = True
    config.or2orn_normalization = True

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.NOISE_MODEL = 'additive'
    config.ORN_NOISE_STD = 0.25

    config.orn2pn_normalization = True
    config.save_every_epoch= True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [0]
    return config, hp_ranges

path = './files_temp/receptor_expression'
t(receptor(), path,s=0,e=100)
#
path = os.path.join(path, '000000', 'epoch')
w_ors = tools.load_pickle(path, 'w_or')
w_orn = tools.load_pickle(path, 'w_orn')
glo_scores = []

for w0, w1 in zip(w_ors, w_orn):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(20, 14))
    axs[0].imshow(w0)
    axs[0].set_title('W_OR')
    axs[1].imshow(w1)
    axs[1].set_title('W_ORN')
    tools.compute_glo_score(w1, 50)
    out = np.matmul(w0, w1)
    ind_max = np.argmax(out, axis=0)
    ind_sort = np.argsort(ind_max)
    out = out[:, ind_sort]
    axs[2].imshow(out)
    axs[2].set_title('Multiplied')
    glo_scores.append(tools.compute_glo_score(out, 50)[0])
    plt.show()
print(glo_scores)

w0 = w_ors[-1]
w1 = w_orn[-1]
n_orn = 50

ind_max = np.argmax(w0, axis=0)
w_out = np.zeros((n_orn, n_orn))
for i in range(n_orn):
    mask = ind_max == i
    masked = w1[mask,:]
    w_out[i,:] = np.mean(masked, axis= 0)

# def temp():
#     config = configs.FullConfig()
#     config.data_dir = './datasets/proto/standard'
#     config.max_epoch = 8
#     config.replicate_orn_with_tiling = True
#     config.N_ORN_DUPLICATION = 10
#     config.NOISE_MODEL = 'multiplicative'
#
#     config.train_pn2kc = False
#     config.sparse_pn2kc = True
#     config.train_kc_bias = True
#     config.kc_loss = False
#     config.initial_pn2kc = 0
#     config.save_every_epoch = False
#
#     config.skip_orn2pn = True
#     config.direct_glo = False
#     config.initializer_pn2kc = 'constant'
#     # Ranges of hyperparameters to loop over
#     hp_ranges = OrderedDict()
#     hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
#     hp_ranges['kc_inputs'] = [3, 7, 11, 15, 20, 30, 40, 50]
#     return config, hp_ranges

# path = './files_temp/vary_noise_kc_input_skip_multiplicative'
# t(temp(), path,s=0,e=100)
# analysis_pn2kc_training.plot_distribution(path)
# analysis_pn2kc_training.plot_sparsity(path)
# analysis_pn2kc_training.plot_pn2kc_initial_value(path)
# analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='ORN_NOISE_STD')
# analysis_pn2kc_training.plot_weight_distribution_per_kc(path,xrange=20, loopkey='ORN_NOISE_STD')
# sa.plot_results(path, x_key='kc_inputs', y_key='val_acc', loop_key='ORN_NOISE_STD')

# wglos = tools.load_pickle(path, 'w_glo')
# for wglo in wglos:
#     wglo= tools._reshape_worn(wglo, 50, mode='tile')
#     wglo = wglo.mean(axis=0)
#     sorted_wglo = np.sort(wglo, axis=0)
#     sorted_wglo = np.flip(sorted_wglo, axis=0)
#     mean = np.mean(sorted_wglo, axis=1)
#     std = np.std(sorted_wglo, axis=1)
#     plt.plot(mean)
    # plt.show()
#
# plt.show()
