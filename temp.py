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
    '''

    '''
    # config = configs.input_ProtoConfig()
    # config.label_type = 'multi_head_sparse'
    # config.has_special_odors = True
    # task.save_proto(config, folder_name='multi_head')

    config = configs.FullConfig()
    config.max_epoch = 12
    config.batch_size = 256
    config.save_every_epoch = True

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.kc_norm_pre = 'batch_norm'

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0.25

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    # config.kc_norm_pre = 'batch_norm'

    config.data_dir = './datasets/proto/standard'

    hp_ranges = OrderedDict()
    hp_ranges['kc_dropout'] = [True]

    return config, hp_ranges

path = './files/temp'
# try:
#     shutil.rmtree(path)
# except:
#     pass
# t(temp(), path, s=0, e=100)

# path_ = os.path.join(path, '000000')
# glo_in, glo_out, kc_out, results = sa.load_activity(path_)
# b_orns = tools.load_pickle(path, 'b_glo')
# plt.hist(glo_in.flatten(), bins=20)
# plt.show()

# w_orns = tools.load_pickle(path, 'w_orn')
# avg_gs, all_gs = tools.compute_glo_score(w_orns[0], 50, mode='tile', w_or = None)
# plt.hist(all_gs, bins=20, range=[0,1])
# sa._easy_save(path, 'hist')

# sa.plot_weights(path, var_name = 'w_or', sort_axis=0, dir_ix=0)
# sa.plot_weights(path, var_name = 'w_orn', sort_axis=1, dir_ix=0)
# sa.plot_weights(path, var_name = 'w_combined', dir_ix=0)


# analysis_multihead.main1(arg='multi_head')
#
# path = './files/metalearn'
analysis_training.plot_distribution(path, xrange=.5, log=False)
analysis_training.plot_distribution(path, xrange=.5, log=True)
# analysis_training.plot_sparsity(path, dynamic_thres=True, thres=.1, visualize=True)
# plt.show()

# epoch_path = './files/metalearn/000001/epoch'
# sa.plot_weights(epoch_path, var_name='w_glo', sort_axis=-1, dir_ix=-1)
# sa.plot_weights(epoch_path, var_name='w_orn', sort_axis=1, dir_ix=-1, average=True)



# analysis_training.plot_distribution(path)
# analysis_training.plot_pn2kc_claw_stats(path, x_key = 'n_trueclass', dynamic_thres=False)
# analysis_activity.sparseness_activity(path, 'glo_out')
# analysis_activity.sparseness_activity(path, 'kc_out')
# analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')


# sa.plot_weights(path, var_name='model/layer3/kernel:0', dir_ix=0)