import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa
import pickle
import model as network_models
import standard.analysis_activity as analysis_activity
import standard.analysis_pn2kc_training as analysis_training
import shutil
import matplotlib.pyplot as plt
import standard.analysis_multihead as analysis_multihead

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

def temp():
    config = configs.FullConfig()
    config.max_epoch = 9
    config.data_dir = './datasets/proto/standard'
    # config.direct_glo = True
    # config.sparse_pn2kc = False
    # config.kc_dropout = False
    # config.train_pn2kc = True
    config.pn_norm_pre = 'batch_norm'
    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0
    config.kc_dropout = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    i = [0, .6]
    # config.datasets = ['./datasets/proto/standard','./datasets/proto/80_20']
    hp_ranges['apl'] = [False]
    return config, hp_ranges

def train_multihead():
    '''

    '''
    # config = configs.input_ProtoConfig()
    # config.label_type = 'multi_head_sparse'
    # config.has_special_odors = True
    # task.save_proto(config, folder_name='multi_head')

    config = configs.FullConfig()
    config.max_epoch = 3
    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    # config.initial_pn2kc = .1
    # config.train_kc_bias = False
    # config.kc_loss = False

    config.pn_norm_pre = 'batch_norm'
    config.data_dir = './datasets/proto/multi_head'
    config.save_every_epoch = True

    hp_ranges = OrderedDict()
    hp_ranges['dummy_var'] = [True]

    return config, hp_ranges

path = './files/multi_head'
try:
    shutil.rmtree(path)
except:
    pass
# t(train_multihead(), path, s=0, e=100)

# analysis_multihead.main()

path = './files/metatrain'
analysis_training.plot_distribution(path, xrange=.5)
analysis_training.plot_sparsity(path, dynamic_thres=False, thres=.03)

epoch_path = './files/metatrain/0/epoch'
sa.plot_weights(epoch_path, var_name='w_glo', sort_axis=-1, dir_ix=-1)
sa.plot_weights(epoch_path, var_name='w_orn', sort_axis=1, dir_ix=-1, average=True)
#
# def plot_weight_change_vs_meta_update_magnitude(path, mat):
#     from standard.analysis import _easy_save
#     def _helper_plot(ax, data, color, label):
#         ax.set_xlabel('Epochs')
#         ax.set_ylabel(label, color=color)
#         ax.plot(np.arange(len(data)), data, color=color)
#         # ax.tick_params(axis='y', labelcolor=color)
#
#     epoch_path = os.path.join(path, '0', 'epoch')
#     lr_ix = 0
#     list_of_wglo = tools.load_pickle(epoch_path, mat)
#     list_of_lr = tools.load_pickle(epoch_path, 'model/lr:0')
#
#     weight_diff = []
#     for i in range(len(list_of_wglo)-1):
#         change = list_of_wglo[i+1] - list_of_wglo[i]
#         change = np.sum(np.abs(change))
#         weight_diff.append(change)
#
#     relevant_lr = []
#     for i in range(0, len(list_of_lr)-1):
#         relevant_lr.append(list_of_lr[i][lr_ix])
#
#     fig = plt.figure(figsize=(3.5, 2))
#     ax = fig.add_axes([0.2, 0.2, 0.6, 0.7])
#     _helper_plot(ax, relevant_lr, 'blue', 'Update Magnitude')
#     ax_ = ax.twinx()
#     _helper_plot(ax_, weight_diff, 'orange', 'Weight Change Magnitude')
#     mat = mat.replace('/', '_')
#     mat = mat.replace(':0', '')
#     _easy_save(path, '_{}_change_vs_learning_rate'.format(mat), pdf=True)
#
# plot_weight_change_vs_meta_update_magnitude(path, 'w_glo')
# plot_weight_change_vs_meta_update_magnitude(path, 'w_orn')
# plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0')

# with open(d, 'rb') as f:
#     dict = pickle.load(f)
#     mat = dict['w_glo']
#     lr = dict['model/lr:0']
#
#
#
#     dist = np.sum(mat, axis=1)
#     print(dist)
#     plt.imshow(np.sort(mat, axis=0)[:,300:600][::-1],cmap='RdBu_r', vmin=-1, vmax=1)
#     plt.axis('tight')
#     plt.show()


# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'glo_out')
# analysis_activity.sparseness_activity(path, 'kc_out')
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')


# sa.plot_weights(path, var_name='model/layer3/kernel:0', dir_ix=0)