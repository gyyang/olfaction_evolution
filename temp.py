"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train
import standard.analysis as sa

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

def temp_norm():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/mask'
    config.max_epoch = 4

    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0
    config.kc_norm_post = 'batch_norm'

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['pn_norm_post'] = [None, 'biology']
    return config, hp_ranges

# t(temp(), path, s=0, e=100)
#
# sa.plot_results(path, x_key='pn_norm_post', y_key='val_acc', loop_key='data_dir')
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