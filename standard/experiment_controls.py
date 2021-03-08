from collections import OrderedDict
import os
import copy

import numpy as np

from configs import FullConfig, SingleLayerConfig
from tools import vary_config
import tools

try:
    import standard.analysis as sa
    import standard.analysis_pn2kc_training as analysis_pn2kc_training
    import standard.analysis_pn2kc_random as analysis_pn2kc_random
    import standard.analysis_orn2pn as analysis_orn2pn
    import standard.analysis_activity as analysis_activity
    import standard.analysis_multihead as analysis_multihead
    import standard.analysis_metalearn as analysis_metalearn
    import analytical.numerical_test as numerical_test
    import analytical.analyze_simulation_results as analyze_simulation_results
    import standard.analysis_nonnegative as analysis_nonnegative
except ImportError as e:
    print(e)

testing_epochs = 12


def control_relabel():
    """Study the impact of relabeling dataset."""
    config = FullConfig()
    config.max_epoch = 100
    config.kc_dropout_rate = 0.

    relabel_class = 100
    true_classes = [100, 200, 500, 1000]

    data_dirs = []
    for true_class in true_classes:
        d = 'relabel_' + str(true_class) + '_' + str(relabel_class)
        data_dirs.append('./datasets/proto/' + d)

    config_ranges = OrderedDict()
    config_ranges['data_dir'] = data_dirs
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_relabel_prune():
    """Standard setting for assessing impact of relabeling datasets.."""
    new_configs = []
    for config in control_relabel():
        config.lr = 2e-4  # made smaller to improve separation
        config.initial_pn2kc = 4./config.N_PN  # for clarity
        config.kc_prune_weak_weights = True
        config.kc_prune_threshold = 1./config.N_PN
        new_configs.append(config)

    return new_configs


def _control_relabel_analysis(path, ax_args=None):
    xkey = 'n_trueclass_ratio'
    ykeys = ['val_acc', 'glo_score', 'K_smart', 'coding_level']

    # Plot network with dropout
    modeldirs = tools.get_modeldirs(path)
    sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys,
                    loop_key='kc_dropout_rate')
    select_dict = {'kc_dropout_rate': 0.}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey)
    sa.plot_xy(modeldirs,
               xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
               ax_args=ax_args)
    sa.plot_results(modeldirs, xkey=xkey, ykey='val_acc')

    select_dict = {'kc_dropout_rate': 0.5}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    analysis_activity.sparseness_activity(modeldirs, 'kc')

    modeldirs = tools.get_modeldirs(path)
    sa.plot_results(modeldirs, xkey=xkey, ykey='val_acc',
                    loop_key='kc_dropout_rate')

    # Plot network trained on relabel task
    select_dict = {'n_trueclass_ratio': 2}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    sa.plot_results(modeldirs, xkey='kc_dropout_rate', ykey=ykeys)
    sa.plot_xy(modeldirs,
               xkey='lin_bins', ykey='lin_hist', legend_key='kc_dropout_rate',
               ax_args=ax_args)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key='kc_dropout_rate')


def control_relabel_analysis(path):
    _control_relabel_analysis(
        path, ax_args={'ylim': [0, 200000], 'xlim': [0, 2.5]})


def control_relabel_prune_analysis(path):
    _control_relabel_analysis(
        path, ax_args={'xlim': [0, 0.5]})


def control_relabel_singlelayer():
    """Training relabel datasets with a single layer network."""
    config = SingleLayerConfig()
    config.max_epoch = 100

    relabel_class = 100
    true_classes = [100, 200, 500, 1000]

    data_dirs = []
    for true_class in true_classes:
        d = 'relabel_' + str(true_class) + '_' + str(relabel_class)
        data_dirs.append('./datasets/proto/' + d)

    config_ranges = OrderedDict()
    config_ranges['data_dir'] = data_dirs

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_relabel_singlelayer_analysis(path):
    xkey = 'n_trueclass_ratio'
    ykeys = 'val_acc'
    sa.plot_results(path, xkey=xkey, ykey=ykeys)
    sa.plot_progress(path, ykeys=ykeys, legend_key=xkey)


def control_nonnegative():
    """Assess impact of non-negativity in ORN-PN weights."""
    config = FullConfig()

    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    config_ranges = OrderedDict()
    config_ranges['sign_constraint_orn2pn'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_nonnegative_receptor():
    configs = control_nonnegative()
    new_configs = list()
    for config in configs:
        config.receptor_layer = True
        config.ORN_NOISE_STD = 0.1
        config.lr = 1e-4  # For receptor, this is the default LR

        # This is the only combination of normalization that works, not sure why
        config.pn_norm_pre = None
        config.kc_norm_pre = 'batch_norm'
        new_configs.append(config)
    return new_configs


def control_nonnegative_analysis(path):
    for sign in [True, False]:
        modeldir = tools.get_modeldirs(path, select_dict={
            'sign_constraint_orn2pn': sign})[0]
        sa.plot_weights(modeldir)
        analysis_orn2pn.plot_distance_distribution(modeldir)

    ykeys = ['val_acc', 'glo_score', 'K_smart']
    sa.plot_progress(path, ykeys=ykeys, legend_key='sign_constraint_orn2pn')
    sa.plot_results(path, xkey='sign_constraint_orn2pn', ykey=ykeys)


def control_standard():
    """Control for standard ORN-PN-KC all trainable model."""
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = np.array([2., 4., 8.])/config.N_PN
    config_ranges['ORN_NOISE_STD'] = [0, 0.1, 0.2]
    config_ranges['kc_prune_weak_weights'] = [False, True]
    # config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_standard_analysis(path):
    default = {'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0., 'lr': 5e-4,
               'train_kc_bias': True, 'initial_pn2kc': 0.08,
               'ORN_NOISE_STD': 0,
               'kc_norm_pre': None,
               }
    ykeys = ['val_acc', 'glo_score', 'K_smart']

    for xkey in default.keys():
        select_dict = copy.deepcopy(default)
        select_dict.pop(xkey)
        modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
        sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys,
                        show_ylabel=(xkey == 'lr'))
        sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey)
        sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist',
                   legend_key=xkey)


def control_pn2kc_prune_boolean():
    """Study impact of PN-KC pruning."""
    config = FullConfig()
    config.max_epoch = 100
    config.data_dir = './datasets/proto/relabel_200_100'
    config.lr = 5e-4
    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    config_ranges = OrderedDict()
    config_ranges['kc_prune_weak_weights'] = [False, True]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_prune_boolean_analysis(path):
    xkey = 'kc_prune_weak_weights'
    ykeys = ['val_acc', 'K_smart']
    sa.plot_progress(path, ykeys=ykeys, legend_key=xkey)
    sa.plot_xy(path, xkey='lin_bins', ykey='lin_hist', legend_key=xkey)


def control_vary_kc():
    """Standard vary the number of KC neurons."""
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.kc_dropout_rate = 0.

    config.lr = 5e-4  # made smaller to improve separation
    config.initial_pn2kc = 4. / config.N_PN  # for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_kc_analysis(path):
    ykeys = ['val_acc', 'glo_score', 'K_smart']
    xticks = [50, 500, 2500, 20000]

    # All networks
    modeldirs = tools.get_modeldirs(path)
    sa.plot_results(modeldirs, xkey='N_KC', ykey=ykeys,
                    loop_key='kc_dropout_rate', ax_args={'xticks': xticks})

    select_dict = {'kc_dropout_rate': 0.}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    sa.plot_results(modeldirs, xkey='N_KC', ykey=ykeys, show_ylabel=True,
                    ax_args={'xticks': xticks}, plot_actual_value=False)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key='N_KC')

    # Example networks
    for n_kc in [50, 10000]:
        select_dict = {'N_KC': n_kc, 'kc_dropout_rate': 0.}
        modeldir = tools.get_modeldirs(path, select_dict=select_dict)[0]
        sa.plot_weights(modeldir)


def _control_vary_pn():
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.kc_dropout_rate = 0.

    config.lr = 5e-4  # made smaller to improve separation
    config.initial_pn2kc = 4. / config.N_PN  # for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    n_pns = [20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    config_ranges = OrderedDict()
    config_ranges['N_PN'] = n_pns
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_pn():
    """Standard vary the number of PN neurons."""
    configs = _control_vary_pn()
    new_configs = list()
    for config in configs:
        config.kc_prune_threshold = 1. / config.N_PN
        config.initial_pn2kc = 4. / config.N_PN
        new_configs.append(config)
    return new_configs


def control_vary_pn_analysis(path):
    ykeys = ['val_acc', 'glo_score', 'K_smart']
    xticks = [20, 50, 200, 1000]

    # All networks
    modeldirs = tools.get_modeldirs(path)
    sa.plot_results(modeldirs, xkey='N_PN', ykey=ykeys,
                    loop_key='kc_dropout_rate', ax_args={'xticks': xticks})

    select_dict = {'kc_dropout_rate': 0.}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    sa.plot_results(modeldirs, xkey='N_PN', ykey=ykeys, show_ylabel=False,
                    ax_args={'xticks': xticks}, plot_actual_value=False)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key='N_PN')

    for n_pn in [30, 50, 200]:
        select_dict = {'N_PN': n_pn, 'kc_dropout_rate': 0.}
        modeldir = tools.get_modeldirs(path, select_dict=select_dict)[0]
        sa.plot_weights(modeldir)

    select_dict = {'N_PN': 200, 'kc_dropout_rate': 0.}
    modeldir = tools.get_modeldirs(path, select_dict=select_dict)[0]
    ix_good, ix_bad = analysis_orn2pn.multiglo_gloscores(
        modeldir, cutoff=.9, shuffle=False)
    analysis_orn2pn.multiglo_pn2kc_distribution(modeldir, ix_good, ix_bad)
    # analysis_orn2pn.multiglo_lesion(modeldir, ix_good, ix_bad)


def vary_orn_corr():
    config = FullConfig()

    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    orn_corrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets = ['./datasets/proto/orn_corr_{:0.2f}'.format(c) for c in orn_corrs]

    config_ranges = OrderedDict()
    config_ranges['orn_corr'] = orn_corrs
    config_ranges['data_dir'] = datasets
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_orn_corr_relabel():
    """Standard setting for varying correlation of ORNs."""
    config = FullConfig()

    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    orn_corrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets = ['./datasets/proto/orn_corr_relabel_{:0.2f}'.format(c) for c in
                orn_corrs]

    config_ranges = OrderedDict()
    config_ranges['orn_corr'] = orn_corrs
    config_ranges['data_dir'] = datasets
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_orn_corr_nosign():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200
    config.pn_norm_pre = 'batch_norm'

    config.batch_size = 8192
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.lr = 2 * 1e-3
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'
    config.sign_constraint_orn2pn = False

    orn_corrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets = ['./datasets/proto/orn_corr_{:0.2f}'.format(c) for c in orn_corrs]

    config_ranges = OrderedDict()
    config_ranges['orn_corr'] = orn_corrs
    config_ranges['data_dir'] = datasets
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_orn_corr_analysis(path):
    xkey = 'orn_corr'
    ykeys = ['val_acc', 'glo_score', 'K_smart']
    sa.plot_results(path, xkey=xkey, ykey=ykeys)
    sa.plot_progress(path, legend_key=xkey, ykeys=ykeys)
    sa.plot_xy(path, xkey='lin_bins', ykey='lin_hist', legend_key=xkey)


def vary_orn_corr_relabel_analysis(path):
    vary_orn_corr_analysis(path)


def _kc_norm():
    """Assesses the effect of KC normalization on glo score and performance"""
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.lr = 5e-4
    config.kc_dropout_rate = 0.
    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    config.kc_norm_pre = None
    config.kc_norm_post = None

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_norm'] = [None, 'batch_norm', 'mean_center',
                                'layer_norm', 'fixed_activity', 'olsen']
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def kc_norm():
    """Assesses the effect of KC normalization on glo score and performance"""
    new_configs = list()
    for c in _kc_norm():
        if c.kc_norm in ['batch_norm', 'layer_norm', 'mean_center']:
            c.kc_norm_pre = c.kc_norm
        if c.kc_norm in ['fixed_activity', 'olsen']:
            c.kc_norm_post = c.kc_norm
        new_configs.append(c)

    return new_configs


def kc_norm_analysis(path):
    select_dict = {'kc_prune_weak_weights': True}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    ykeys = ['val_acc', 'K_smart', 'glo_score']
    xkey = 'kc_norm'
    sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys,
                    loop_key='kc_dropout_rate',
                    figsize=[2.5, 1.2 + 0.7 * (3 - 1)])

    select_dict.update({'kc_dropout_rate': 0.5})
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey)
    sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
               ax_args={'xlim': [0, 1], 'ylim': [0, 200000]})


######################## Currently unused experiments #######################


def vary_kc_claws():
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence

    TODO: Not in manuscript, remove in future
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.train_pn2kc = False

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm' #not necessary, but for standardization

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_inputs'] = list(range(1,15, 2)) + list(range(15,30, 3)) + \
                             list(range(30, 50, 5))
    config_ranges['ORN_NOISE_STD'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_kc_claws_analysis(path):
    t = [1, 2, 9, 19, 29, 39, 49, 59, 69]
    for i in t:
        res = tools.load_all_results(path, argLast=False, ix=i)
        sa.plot_results(path, xkey='kc_inputs', ykey='log_val_loss',
                        select_dict={'ORN_NOISE_STD': 0}, res=res,
                        string=str(i), figsize=(2, 2))

    sa.plot_progress(path, select_dict={'kc_inputs': [7, 15, 30],
                                        'ORN_NOISE_STD': 0},
                     legends=['7', '15', '30'])
    # analysis_activity.sparseness_activity(path, 'kc')
    # import tools
    # for i in range(8):
    #     res = tools.load_all_results(path, argLast=False, ix=i)
    #     sa.plot_results(path, xkey='kc_inputs', ykey='train_loss',
    #                     select_dict={'ORN_NOISE_STD':0}, res=res, string = str(i))

    # sa.plot_results(path, xkey='kc_inputs', ykey='val_acc', loop_key='ORN_NOISE_STD',
    #                 figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
    sa.plot_results(path, xkey='kc_inputs', ykey='val_acc',
                    select_dict={'ORN_NOISE_STD': 0},
                    figsize=(2, 2))
    # sa.plot_results(path, xkey='kc_inputs', ykey='log_val_loss', loop_key='ORN_NOISE_STD',
    #                 figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
    #                 ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})
    sa.plot_results(path, xkey='kc_inputs', ykey='log_val_loss',
                    select_dict={'ORN_NOISE_STD': 0},
                    figsize=(2, 2),
                    ax_args={'ylim': [-1, 2], 'yticks': [-1, 0, 1, 2]})


def train_claw_configs():
    """
    Train (with or without loss) or fix connections from PN2KC while skipping ORN2PN
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7

    TODO: Not in manuscript, remove in future
    """
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['train_pn2kc'] = [True, False]
    config_ranges['sparse_pn2kc'] = [False, True]
    config_ranges['train_kc_bias'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential')
    return configs



def controls_receptor():
    """Control for receptor network.

    TODO: remain to be added to manuscript
    """
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config.train_pn2kc = False

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.pn_norm_pre = 'batch_norm'
    config.ORN_NOISE_STD = .4

    config.replicate_orn_with_tiling = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30,100]
    config_ranges['or2orn_normalization'] = [False, True]
    config_ranges['pn_norm_pre'] = ['None', 'batch_norm']

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def controls_receptor_analysis(path):
    default = {'N_ORN_DUPLICATION': 10, 'or2orn_normalization': True,
               'pn_norm_pre': 'batch_norm'}
    sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='or_glo_score',
                    select_dict={'or2orn_normalization': True,
                                 'pn_norm_pre': 'batch_norm'}),
    sa.plot_results(path, xkey='or2orn_normalization', ykey='or_glo_score',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'pn_norm_pre': 'batch_norm'})
    sa.plot_results(path, xkey='pn_norm_pre', ykey='or_glo_score',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'or2orn_normalization': True})

    sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='combined_glo_score',
                    select_dict={'or2orn_normalization': True,
                                 'pn_norm_pre': 'batch_norm'}),
    sa.plot_results(path, xkey='or2orn_normalization',
                    ykey='combined_glo_score',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'pn_norm_pre': 'batch_norm'})
    sa.plot_results(path, xkey='pn_norm_pre', ykey='combined_glo_score',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'or2orn_normalization': True})

    sa.plot_results(path, xkey='N_ORN_DUPLICATION', ykey='val_acc',
                    select_dict={'or2orn_normalization': True,
                                 'pn_norm_pre': 'batch_norm'}),
    sa.plot_results(path, xkey='or2orn_normalization', ykey='val_acc',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'pn_norm_pre': 'batch_norm'})
    sa.plot_results(path, xkey='pn_norm_pre', ykey='val_acc',
                    select_dict={'N_ORN_DUPLICATION': 10,
                                 'or2orn_normalization': True})


def vary_init_sparse():
    """Vary if initialization is dense or sparse.

    TODO: Remain to be added to manuscript
    """
    config = FullConfig()
    config.max_epoch = 100
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    config.pn_norm_pre = 'batch_norm'

    config.data_dir = './datasets/proto/standard'
    config.save_every_epoch = True
    config_ranges = OrderedDict()
    config_ranges['initializer_pn2kc'] = ['constant', 'single_strong']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_apl():
    """Vary APL.

    TODO: Remain to be added to manuscript
    """
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    config.save_every_epoch = True

    config_ranges = OrderedDict()
    config_ranges['apl'] = [False, True]
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_apl_analysis(path):
    analysis_activity.sparseness_activity(
        path, 'kc', activity_threshold=0., lesion_kwargs=None)
    lk = {'name': 'model/apl2kc/kernel:0',
          'units': 0, 'arg': 'outbound'}
    analysis_activity.sparseness_activity(
        path, 'kc', activity_threshold=0., lesion_kwargs=lk,
        figname='lesion_apl_')


def control_pn2kc_inhibition():
    """Assess impact of APL with recurrent inhibition.

    TODO: Remain to be added to manuscipt.
    """
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.kc_dropout_rate = 0.

    config.initial_pn2kc = 4. / config.N_PN
    config.kc_prune_threshold = 1. / config.N_PN
    config.kc_recinh = True
    config.kc_recinh_step = 10
    config.kc_prune_weak_weights = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_recinh_coeff'] = list(np.arange(0, 1.01, 0.2))
    # config_ranges['kc_recinh_step'] = list(range(1, 10, 2))
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_inhibition_analysis(path):
    xkey = 'kc_recinh_coeff'
    ykeys = ['val_acc', 'glo_score', 'K_inferred', 'coding_level']
    loop_key = None
    select_dict = {'kc_prune_weak_weights': True, 'kc_recinh_step': 10}

    modeldirs = tools.get_modeldirs(
        path, select_dict=select_dict, acc_min=0.5)

    sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys, loop_key=loop_key)
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey)

    sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist', legend_key=xkey)


def control_orn2pn_random():
    """Assessing the randomness of ORN-PN connections.

    TODO: Remain to be added to manuscript
    """
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 25
    config.N_ORN_DUPLICATION = 1
    config.pn_norm_pre = 'batch_norm'
    config.train_orn2pn = False
    config.orn_manual = True

    config.train_pn2kc = True
    config.sparse_pn2kc = False

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    alpha_values = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    l = len(alpha_values)
    config_ranges['orn_random_alpha'] = alpha_values * 2
    config_ranges['train_pn2kc'] = [True] * l + [False] * l
    config_ranges['sparse_pn2kc'] = [False] * l + [True] * l
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def control_orn2pn_random_analysis(path):
    xks = ['orn_random_alpha', 'glo_score']
    ykeys = ['val_acc']
    trainable_dict = {'train_pn2kc': True, 'sparse_pn2kc': False}
    fixed_dict = {'train_pn2kc': False, 'sparse_pn2kc': True}

    for yk in ykeys:
        for xk in xks:
            for d in [trainable_dict, fixed_dict]:
                sa.plot_results(path,
                                xkey=xk,
                                ykey=yk,
                                select_dict=d,
                                ax_args={'xticks': np.arange(0, 1.01, 0.2)},
                                plot_actual_value=False)

                sa.plot_progress(path,
                                 select_dict=d,
                                 ykeys=[yk],
                                 legend_key='orn_random_alpha')

    sa.plot_results(path, xkey='orn_random_alpha', ykey='glo_score')


def control_nonnegative_full():
    """Assess impact of non-negativity in weights."""
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_100_100'

    config.kc_dropout_rate = 0.
    config.kc_prune_weak_weights = False

    config_ranges = OrderedDict()
    config_ranges['sign_constraint_orn2pn'] = [True, False]
    config_ranges['sign_constraint_pn2kc'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs