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

def temp_norm():
    config = configs.FullConfig()
    config.max_epoch = 6

    config.direct_glo = True
    # config.kc_dropout = False
    # config.sparse_pn2kc = False
    # config.train_pn2kc = True

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    i = [0, .6]
    datasets = ['./datasets/proto/concentration_mask_row_' + str(s) for s in i]
    # datasets = ['./datasets/proto/standard','./datasets/proto/concentration_mask_row_0']
    hp_ranges['data_dir'] = ['./datasets/proto/standard'] + ['./datasets/proto/concentration'] + datasets
    # hp_ranges['data_dir'] = [''datasets/proto/concentration_mean_mask']
    hp_ranges['pn_norm_pre'] = ['biology']
    return config, hp_ranges
#
path = './files_temp/temp'
try:
    shutil.rmtree(path)
except:
    pass
t(temp_norm(), path, s=0, e=100)

# sa.plot_results(path, x_key='data_dir', y_key='val_acc', loop_key='pn_norm_pre',
#                 select_dict={
#                     'pn_norm_pre': ['None', 'fixed_activity'],
#                     'data_dir': ['./datasets/proto/standard',
#                                  './datasets/proto/concentration',
#                                  './datasets/proto/concentration_mask_row_0'
#                                  ]
#                 }, sort=False)
#
# sa.plot_results(path, x_key='data_dir', y_key='val_acc', loop_key='pn_norm_pre',
#                 select_dict={
#                     'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
#                     'data_dir': ['./datasets/proto/concentration_mask_row_0',
#                                  './datasets/proto/concentration_mask_row_0.6',
#                                  ]
#                 })

rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
rho = tools.load_pickle(path, 'model/layer1/rho:0')
m = tools.load_pickle(path, 'model/layer1/m:0')
print(rmax)
print(rho)
print(m)

# analysis_training.plot_sparsity(path, dynamic_thres=True)
# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'glo_out')
# analysis_activity.sparseness_activity(path, 'kc_out')
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')


# sa.plot_weights(path, var_name='model/layer3/kernel:0', dir_ix=0)