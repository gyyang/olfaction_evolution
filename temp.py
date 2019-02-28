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
#     config.max_epoch = 10
#
#     config.direct_glo = True
#     config.save_every_epoch = True
#     # config.sparse_pn2kc = False
#     # config.train_pn2kc = True
#
#     config.replicate_orn_with_tiling = False
#     config.N_ORN_DUPLICATION = 1
#     config.ORN_NOISE_STD = 0
#     # config.kc_loss = True
#     config.pn_norm_pre = 'batch_norm'
#
#     # Ranges of hyperparameters to loop over
#     hp_ranges = OrderedDict()
#     # hp_ranges['data_dir'] = ['./datasets/proto/standard', './datasets/proto/combinatorial']
#     hp_ranges['data_dir'] = ['./datasets/proto/combinatorial']
#     # hp_ranges['pn_norm_pre'] = ['None', 'biology']
#     return config, hp_ranges
#
# path = './files_temp/temp'
# shutil.rmtree(path)
# t(temp_norm(), path, s=0, e=100)



# analysis_training.plot_sparsity(path, dynamic_thres=True)
# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'glo_out')
# analysis_activity.sparseness_activity(path, 'kc_out')
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')

# sa.plot_results(path, x_key='skip_pn2kc', y_key='val_acc', loop_key='N_CLASS')
# sa.plot_weights(path, var_name='model/layer3/kernel:0', dir_ix=0)