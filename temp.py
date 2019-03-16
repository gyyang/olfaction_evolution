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
    config.kc_dropout_rate = .2

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
t(temp(), path, s=0, e=100)

# analysis_multihead.main()

# path = './files/metatrain/valence_peter'
# analysis_training.plot_distribution(path)
# analysis_training.plot_sparsity(path, dynamic_thres=False)

# epoch_path = './files/metatrain/valence_peter/0/epoch'
# sa.plot_weights(epoch_path, var_name='w_glo', sort_axis=-1, dir_ix=-1)
# sa.plot_weights(epoch_path, var_name='w_orn', sort_axis=1, dir_ix=-1)

# d = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\files\metatrain\valence_peter\0\epoch\3200\model.pkl'
# with open(d, 'rb') as f:
#     dict = pickle.load(f)
#     mat = dict['w_glo']
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