"""Experiments and corresponding analysis.

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

Analysis functions by convention are named
def name_analysis()
"""

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


def metalearn():
    config = MetaConfig()
    config.meta_lr = .001
    config.N_CLASS = 10 #10
    config.save_every_epoch = True
    config.meta_batch_size = 32 #32
    config.meta_num_samples_per_class = 8 #16
    config.meta_print_interval = 500

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.output_max_lr = 2.0 #2.0

    config.metatrain_iterations = 15000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    config.initial_pn2kc = 0.05 #0.05

    config.data_dir = './datasets/proto/meta_dataset'

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def metalearn_analysis(path):
    import os
    path = os.path.join(path, '000000','epoch','1000')
    sa.plot_weights(path, var_name='w_orn', sort_axis=1, average=False)
    sa.plot_weights(path, var_name='w_glo', sort_axis=-1)
    analysis_pn2kc_training.plot_distribution(path, xrange=1)
    analysis_pn2kc_training.plot_sparsity(path)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)


def meta_standard():
    config = MetaConfig()
    # config.data_dir = './datasets/proto/relabel_200_100'
    config.data_dir = './datasets/proto/standard'
    config.kc_dropout = False
    config.kc_dropout_rate = 0.
    config.kc_prune_weak_weights = True

    config.replicate_orn_with_tiling = True
    # config.N_ORN_DUPLICATION = 1
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    # config.skip_orn2pn = True

    config.meta_lr = 5e-4
    config.N_CLASS = 2
    config.meta_batch_size = 32
    config.meta_num_samples_per_class = 16
    config.meta_print_interval = 100
    config.output_max_lr = 2.0
    config.meta_update_lr = .2
    config.metatrain_iterations = 10000
    config.meta_trainable_lr = True

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


def meta_vary_or(n_pn=50):
    """Training networks with different number of PNs and vary hyperparams."""
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
    # config.data_dir = './datasets/proto/standard'

    config.skip_orn2pn = True
    config.meta_trainable_lr = True

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


def meta_vary_or_analysis(path, n_pn=None):
    def _meta_vary_or_analysis(path, n_pn):
        # Analyze individual network
        select_dict = {'N_PN': n_pn}
        modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
        _modeldirs = modeldirs
        sa.plot_progress(_modeldirs, ykeys=['val_acc', 'K_smart'],
                         legend_key='meta_lr')

        _modeldirs = modeldirs
        sa.plot_xy(_modeldirs, xkey='lin_bins', ykey='lin_hist',
                   legend_key='meta_lr')

        sa.plot_results(_modeldirs, xkey='meta_lr', ykey=['val_acc', 'K_smart'])

    if n_pn is not None:
        _meta_vary_or_analysis(path, n_pn)

    else:
        import glob
        path = path + '_pn'  # folders named XX_pn50, XX_pn100, ..
        folders = glob.glob(path + '*')
        n_orns = sorted([int(folder.split(path)[-1]) for folder in folders])
        Ks = list()
        for n_orn in n_orns:
            _path = path + str(n_orn)
            # _meta_vary_or_analysis(_path, n_pn=n_orn)

            # NOTICE this is chosen after manual inspection
            # TODO: Need an automated method
            select_dict = {'meta_lr': 5e-4}
            modeldirs = tools.get_modeldirs(_path, select_dict=select_dict)

            # modeldirs = tools.filter_modeldirs(
            #     modeldirs, exclude_badkc=True, exclude_badpeak=True)

            modeldirs = tools.sort_modeldirs(modeldirs, 'meta_lr')
            modeldirs = [modeldirs[-1]]  # Use model with highest LR
            # modeldirs = [modeldirs[0]]  # Use model with lowest LR

            res = tools.load_all_results(modeldirs)
            Ks.append(res['K_smart'])

        for plot_dim in [False, True]:
            analysis_pn2kc_training.plot_all_K(n_orns, Ks, plot_box=True,
                                               plot_dim=plot_dim,
                                               path=path)


def meta_vary_or_prune_analysis(path, n_pn=None):
    meta_vary_or_analysis(path, n_pn=n_pn)


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
