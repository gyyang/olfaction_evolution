"""Experiments and corresponding analysis.

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

Analysis functions by convention are named
def name_analysis()
"""

import copy
from collections import OrderedDict

from configs import MetaConfig
from tools import vary_config
import tools
import settings

try:
    import standard.analysis as sa
    import standard.analysis_pn2kc_training as analysis_pn2kc_training
    import standard.analysis_pn2kc_random as analysis_pn2kc_random
    import standard.analysis_orn2pn as analysis_orn2pn
    import standard.analysis_rnn as analysis_rnn
    import standard.analysis_activity as analysis_activity
    import standard.analysis_multihead as analysis_multihead
except ImportError as e:
    print(e)


use_torch = settings.use_torch


def _meta_standard_config(config):
    """Put here instead of default config for clarity."""
    config.data_dir = './datasets/proto/standard'
    config.kc_dropout = False
    config.kc_dropout_rate = 0.
    config.kc_prune_weak_weights = True

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 1
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    config.skip_orn2pn = True

    config.meta_lr = 5e-4
    config.N_CLASS = 2
    config.meta_batch_size = 32
    config.meta_num_samples_per_class = 16
    config.meta_print_interval = 100
    config.output_max_lr = 2.0
    config.meta_update_lr = .2
    config.metatrain_iterations = 10000
    config.meta_trainable_lr = True
    return config


def meta_standard():
    config = MetaConfig()
    config = _meta_standard_config(config)

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [0]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def meta_standard_analysis(path):
    modeldirs = tools.get_modeldirs(path)
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_smart'],
                     legend_key='lr')
    modeldir = modeldirs[0]
    sa.plot_weights(modeldir)
    analysis_pn2kc_training.plot_distribution(modeldir, xrange=0.5)
    analysis_pn2kc_training.plot_sparsity(modeldir, epoch=-1)

    # analysis_activity.distribution_activity(path, ['glo', 'kc'])
    # analysis_activity.sparseness_activity(path, ['glo', 'kc'])


def meta_control_standard():
    config = MetaConfig()
    config = _meta_standard_config(config)

    config_ranges = OrderedDict()
    config_ranges['meta_lr'] = [1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['N_CLASS'] = [2, 3, 4]
    config_ranges['meta_update_lr'] = [.1, .2, .5, 1.0]
    config_ranges['meta_num_samples_per_class'] = [4, 8, 16, 32]
    config_ranges['kc_dropout_rate'] = [0., 0.25, 0.5, 0.75]
    config_ranges['kc_prune_weak_weights'] = [True, False]
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']
    config_ranges['skip_orn2pn'] = [True, False]
    config_ranges['data_dir'] = ['./datasets/proto/standard',
                                 './datasets/proto/relabel_200_100']

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def meta_control_standard_analysis(path):
    default = {'meta_lr': 5e-4, 'N_CLASS': 2, 'meta_update_lr': .2,
               'meta_num_samples_per_class': 16, 'kc_dropout_rate': 0.,
               'kc_prune_weak_weights': True, 'pn_norm_pre': 'batch_norm',
               'kc_norm_pre': 'batch_norm', 'skip_orn2pn': True,
               'data_dir': './datasets/proto/standard'
               }
    ykeys = ['val_acc', 'K_smart']

    for xkey in default.keys():
        select_dict = copy.deepcopy(default)
        select_dict.pop(xkey)
        show_ylabel = xkey == 'meta_lr'
        modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
        sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys,
                        show_ylabel=show_ylabel)
        sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey,
                         show_ylabel=show_ylabel)
        sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist',
                   legend_key=xkey)


def meta_vary_or(n_pn=50):
    """Standard settings for different number of PNs in meta-learning."""
    config = MetaConfig()
    config = _meta_standard_config(config)
    config.metatrain_iterations = 15000  # Train a bit longer

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config_ranges = OrderedDict()
    config_ranges['meta_lr'] = [1e-3, 5e-4, 2e-4, 1e-4]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def meta_vary_or_prune(n_pn=50):
    new_configs = []
    for config in meta_vary_or(n_pn=n_pn):
        config.N_CLASS = 2
        config.kc_prune_weak_weights = True
        new_configs.append(config)
    return new_configs


def meta_vary_or_analysis(path):
    def _vary_or_analysis(path, legend_key):
        modeldirs = tools.get_modeldirs(path)
        sa.plot_progress(modeldirs, ykeys=['val_acc', 'K_smart'],
                         legend_key=legend_key)
        sa.plot_results(modeldirs, xkey=legend_key,
                        ykey=['val_acc', 'K_smart'])
        sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist',
                   legend_key=legend_key)

    import glob
    _path = path + '_pn'  # folders named XX_pn50, XX_pn100, ..
    folders = glob.glob(_path + '*')
    n_orns = sorted([int(folder.split(_path)[-1]) for folder in folders])
    for n_orn in n_orns:
        _vary_or_analysis(_path + str(n_orn), legend_key='meta_lr')

    analysis_pn2kc_training.plot_all_K(path)


def meta_vary_or_prune_analysis(path, n_pn=None):
    meta_vary_or_analysis(path, n_pn=n_pn)


def meta_trainable_lr():
    config = MetaConfig()
    config.meta_lr = .001
    config.N_CLASS = 5 #10
    config.save_every_epoch = False
    config.meta_batch_size = 32 #32
    config.meta_num_samples_per_class = 16 #16
    config.meta_print_interval = 100

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 1
    config.output_max_lr = 2.0 #2.0
    config.meta_update_lr = .2
    config.prune = False

    config.metatrain_iterations = 10000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'

    config.kc_dropout = False

    # config.data_dir = './datasets/proto/meta_dataset'
    config.data_dir = './datasets/proto/standard'

    config.skip_orn2pn = True
    config.meta_trainable_lr = True

    config_ranges = OrderedDict()
    config_ranges['meta_trainable_lr'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def meta_trainable_lr_analysis(path):
    modeldirs = tools.get_modeldirs(path)
    # accuracy
    ykeys = ['val_acc', 'train_post_acc', 'glo_score', 'K_smart']
    sa.plot_progress(modeldirs, ykeys=ykeys,
                     legend_key='meta_trainable_lr')


def meta_num_updates():
    config = MetaConfig()
    config.meta_lr = .001
    # config.N_CLASS = 5 #10
    config.N_CLASS = 2
    config.save_every_epoch = False
    config.meta_batch_size = 32 #32
    config.meta_num_samples_per_class = 16 #16
    config.meta_print_interval = 100

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 1
    config.output_max_lr = 10. #2.0
    config.meta_update_lr = .1
    config.kc_prune_weak_weights = True

    config.metatrain_iterations = 10000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'

    config.kc_dropout = False

    # config.data_dir = './datasets/proto/meta_dataset'
    config.data_dir = './datasets/proto/standard'

    config.skip_orn2pn = True
    config.meta_trainable_lr = True

    n_pn = 50
    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)

    config_ranges = OrderedDict()
    config_ranges['meta_num_updates'] = [1]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs
