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

def temp1():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/200_20'
    config.max_epoch = 20

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    # config.train_kc_bias = False
    # config.train_pn2kc = True
    # config.sparse_pn2kc = False
    # config.save_every_epoch = True
    # config.kc_norm_pre = 'batch_norm'

    hp_ranges = OrderedDict()
    hp_ranges['extra_layer'] = [False, True]
    return config, hp_ranges


def basic():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 10

    config.N_ORN_DUPLICATION = 10
    config.replicate_orn_with_tiling = True
    config.direct_glo = True

    # config.pn_norm_pre = 'batch_norm'
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.initial_pn2kc = .1

    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [0]
    return config, hp_ranges

path = './files_temp/movie_kc'
shutil.rmtree(path)
t(basic(), path, s=0, e=100)
analysis_training.plot_sparsity(path, dynamic_thres=True)
# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'kc_out')
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')

# sa.plot_results(path, x_key='extra_layer', y_key='val_acc')

