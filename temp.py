"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa
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


def temp_oracle():
    config = configs.FullConfig()
    config.max_epoch = 10
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

path = './files_temp/relabel_layers'
# t(temp_oracle(), path, s=0, e=100)
sa.plot_results(path, x_key='n_trueclass', y_key='val_acc', loop_key='model', sort=False)
# sa.plot_results(path, x_key='data_dir', y_key='val_loss', loop_key='model', sort=False)
# sa.plot_results(path, x_key='data_dir', y_key='train_loss', loop_key='model', sort=False)
analysis_activity.sparseness_activity(path, 'kc_out')

# sa.plot_results(path, x_key='pn_norm_post', y_key='glo_score', loop_key='data_dir')
#
# try:
#     rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
#     print('rmax: {}'.format(rmax))
#     rho = tools.load_pickle(path, 'model/layer1/rho:0')
#     print('rho: {}'.format(rho))
#     m = tools.load_pickle(path, 'model/layer1/m:0')
#     print('m: {}'.format(m))
# except:
#     pass
#
# try:
#     gamma = tools.load_pickle(path, 'model/layer1/LayerNorm/gamma:0')
#     print('gamma params: {}'.format(gamma))
# except:
#     pass