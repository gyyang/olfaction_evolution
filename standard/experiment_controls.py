from collections import OrderedDict
import os
import copy

import numpy as np

from configs import FullConfig
from tools import vary_config
import tools

try:
    import standard.analysis as sa
    import standard.analysis_pn2kc_peter
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
    """Standard training setting"""
    config = FullConfig()
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


def control_relabel_prune():
    """Control for standard ORN-PN-KC all trainable model with pruning."""
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
    ykeys = ['coding_level', 'glo_score', 'val_acc', 'K_smart']
    sa.plot_results(path, xkey=xkey, ykey=ykeys)
    sa.plot_progress(path, ykeys=ykeys, legend_key=xkey)
    sa.plot_xy(path,
               xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
               ax_args=ax_args)
    analysis_activity.sparseness_activity(path, 'kc')


def control_relabel_analysis(path):
    _control_relabel_analysis(
        path, ax_args={'ylim': [0, 200], 'xlim': [0, 2.5]})


def control_relabel_prune_analysis(path):
    _control_relabel_analysis(
        path, ax_args={'ylim': [0, 80], 'xlim': [0, 0.5]})


def control_nonnegative():
    """Standard training setting"""
    config = FullConfig()
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'
    config.data_dir = './datasets/proto/standard'

    config_ranges = OrderedDict()
    config_ranges['sign_constraint_orn2pn'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_nonnegative_analysis(path):
    sa.plot_weights(os.path.join(path, '000000'), sort_axis=1, average=False)
    sa.plot_weights(os.path.join(path, '000001'), sort_axis=1, average=False,
                    positive_cmap=False, vlim=[-1, 1])
    for ix in range(0, 2):
        standard.analysis_orn2pn.correlation_matrix(path, ix=ix, arg='ortho')
        standard.analysis_orn2pn.correlation_matrix(path, ix=ix, arg='corr')

    # # #sign constraint
    sa.plot_progress(path, ykeys=['glo_score', 'val_acc'],
                     legend_key='sign_constraint_orn2pn')
    sa.plot_results(path, xkey='sign_constraint_orn2pn', ykey='glo_score')
    sa.plot_results(path, xkey='sign_constraint_orn2pn', ykey='val_acc')


def control_standard():
    """Control for standard ORN-PN-KC all trainable model."""
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.initial_pn2kc = 4./config.N_PN  # necessary for analysis

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = np.array([2., 4., 8.])/config.N_PN
    config_ranges['ORN_NOISE_STD'] = [0, 0.1, 0.2]
    # config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_standard_analysis(path):
    default = {'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5, 'lr': 1e-3,
               'train_kc_bias': True, 'initial_pn2kc': 0.08,
               'ORN_NOISE_STD': 0,
               'kc_norm_pre': None,
               }
    ykeys = ['glo_score', 'val_acc', 'K_smart']

    for xkey in default.keys():
        select_dict = copy.deepcopy(default)
        select_dict.pop(xkey)
        modeldirs = tools.get_modeldirs(
            path, select_dict=select_dict, acc_min=0.5)

        _modeldirs = analysis_pn2kc_training.filter_modeldirs(
            modeldirs, exclude_badkc=True, exclude_badpeak=True)
        sa.plot_results(_modeldirs, xkey=xkey, ykey=ykeys)
        sa.plot_progress(_modeldirs, ykeys=ykeys, legend_key=xkey)

        _modeldirs = modeldirs
        sa.plot_xy(_modeldirs,
                   xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
                   ax_args={'ylim': [0, 200], 'xlim': [0, 2.5]})


def control_standard_prune():
    """Control for standard ORN-PN-KC all trainable model with pruning."""
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.initial_pn2kc = 4./config.N_PN  # necessary for analysis

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./config.N_PN

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['kc_prune_threshold'] = np.array([0.5, 1., 2.])/config.N_PN
    config_ranges['initial_pn2kc'] = np.array([2., 4., 8.])/config.N_PN

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_orn2pn():
    '''
    '''
    # TODO: to be removed
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.pn_norm_pre = 'batch_norm'

    config.train_pn2kc = False

    # New settings
    config.batch_size = 256  # Much bigger batch size
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 1e-3

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30]
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['ORN_NOISE_STD']= [0, .1, .2]
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_orn2pn_analysis(path):
    default = {'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm',
               'kc_dropout_rate': 0.5, 'N_ORN_DUPLICATION': 10, 'lr': 1e-3}
    ykeys = ['glo_score', 'val_acc']

    for yk in ykeys:
        for xk, v in default.items():
            temp = copy.deepcopy(default)
            temp.pop(xk)
            if xk == 'lr':
                logx = True
            else:
                logx = False
            sa.plot_results(path, xkey=xk, ykey=yk,
                            select_dict=temp, logx=logx)

            sa.plot_progress(path, select_dict=temp, ykeys=[yk], legend_key=xk)


def control_orn2pn_random():
    '''
    '''
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



def control_pn2kc_backup():
    '''
    This is the setup Peter last used
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = [0.05, 0.1, 0.2, 0.5]
    config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_pn2kc():
    '''
    New setup Robert using for torch models
    '''
    # TODO: To be removed
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200

    config.N_ORN_DUPLICATION = 1
    config.direct_glo = True  # skip_orn2pn has same effect
    config.pn_norm_pre = 'batch_norm'

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10./config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = np.array([2., 5., 10., 20.])/config.N_PN
    # config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_pn2kc_analysis(path):
    # TODO: To be removed
    default = {'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5, 'lr': 1e-3}
    ykeys = ['val_acc', 'K_inferred']

    for yk in ykeys:
        exclude_dict = None
        if yk in ['K_inferred', 'sparsity_inferred', 'K', 'sparsity']:
            exclude_dict = {'lr': [3e-3, 1e-2, 3e-2]}

        for xk, v in default.items():
            temp = copy.deepcopy(default)
            temp.pop(xk)
            if xk == 'lr':
                logx = True
            else:
                logx = False
            sa.plot_results(path, xkey=xk, ykey=yk,
                            select_dict=temp, logx=logx)

            sa.plot_progress(path, select_dict=temp, ykeys=[yk],
                             legend_key=xk, exclude_dict=exclude_dict)
    #
    res = standard.analysis_pn2kc_peter.do_everything(path, filter_peaks=False,
                                                      redo=True)
    for xk, v in default.items():
        temp = copy.deepcopy(default)
        temp.pop(xk)
        sa.plot_xy(path, select_dict=temp, xkey='lin_bins_', ykey='lin_hist_',
                   legend_key=xk, log=res,
                   ax_args={'ylim': [0, 500]})


def control_pn2kc_inhibition():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.initial_pn2kc = 4. / config.N_PN
    # config.kc_prune_threshold = 1. / config.N_PN
    config.kc_recinh = True
    config.kc_recinh_step = 10

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    # config_ranges['kc_prune_weak_weights'] = [True, False]
    config_ranges['kc_recinh_coeff'] = list(np.arange(0, 1.01, 0.2))
    # config_ranges['kc_recinh_step'] = list(range(1, 10, 2))
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_inhibition_analysis(path):
    xkey = 'kc_recinh_coeff'
    ykeys = ['val_acc', 'K_inferred']
    # loop_key = 'kc_recinh_step'
    loop_key = None
    select_dict = {'kc_prune_weak_weights': False, 'kc_recinh_step': 9}

    modeldirs = tools.get_modeldirs(
        path, select_dict=select_dict, acc_min=0.5)

    sa.plot_results(modeldirs, xkey=xkey, ykey=ykeys, loop_key=loop_key,
                    figsize=(2.0, 1.2))
    sa.plot_progress(modeldirs, ykeys=ykeys, legend_key=xkey)

    sa.plot_xy(modeldirs, xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
               ax_args={'ylim': [0, 500]})


def control_pn2kc_prune_boolean(n_pn=50):
    """Control pruning."""
    config = FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./n_pn

    # Heuristics
    if n_pn > 50:
        config.lr = 1e-4
    else:
        config.lr = 1e-3

    # This is important for quantitative result
    config.N_KC = min(40000, n_pn ** 2)

    config_ranges = OrderedDict()
    config_ranges['kc_prune_weak_weights'] = [False, True]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_prune_boolean_analysis(path, n_pns=None):
    xkey = 'kc_prune_weak_weights'
    ykeys = ['val_acc', 'K_smart']
    if n_pns is None:
        n_pns = [50, 200]

    for n_pn in n_pns:
        cur_path = path + '_pn' + str(n_pn)
        sa.plot_progress(cur_path, ykeys=ykeys, legend_key=xkey)
        sa.plot_xy(cur_path, xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
                   ax_args={'ylim': [0, n_pn ** 2.4 / 50],
                            'xlim': [0, 8 / n_pn**0.6]})


def control_vary_kc_prune(n_pn=50):
    """Vary KC with pruning, train only PN-KC."""
    config = FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./n_pn

    # Heuristics
    if n_pn > 50:
        config.lr = 1e-4
    else:
        config.lr = 1e-3

    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [int(n**2) for n in np.linspace(50, n_pn, 5)]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_kc_prune_analysis(path, n_pns=None):
    ykeys = ['val_acc', 'K_smart']
    n_pns = n_pns or [200]
    for n_pn in n_pns:
        cur_path = path + '_pn' + str(n_pn)
        sa.plot_progress(cur_path, legend_key='N_KC', ykeys=ykeys)
        sa.plot_results(cur_path, xkey='N_KC', ykey=ykeys,
                        logx=True, figsize=(2.5, 1.5))


def control_vary_kc():
    """Vary KC without pruning, train all."""
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_kc_analysis(path):
    ykeys = ['val_acc', 'glo_score']
    xticks = [50, 200, 1000, 2500, 10000]

    sa.plot_results(path, xkey='N_KC', ykey=ykeys, loop_key='kc_dropout_rate',
                    logx=True, ax_args={'xticks': xticks}, figsize=(2.5, 1.5))


def control_vary_pn():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config_ranges = OrderedDict()
    config_ranges['N_PN'] = [20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_pn_analysis(path):
    # TODO: bring back the analysis
    # sa.plot_weights(os.path.join(path, '000004'), sort_axis=1, average=False)
    # sa.plot_weights(os.path.join(path, '000010'), sort_axis=1, average=False,
    #                 vlim=[0, 5])
    # sa.plot_weights(os.path.join(path, '000022'), sort_axis=1, average=False,
    #                 vlim=[0, 5])
    #
    # ix = 22
    # ix_good, ix_bad = analysis_orn2pn.multiglo_gloscores(path, ix, cutoff=.9,
    #                                                      shuffle=False)
    # analysis_orn2pn.multiglo_pn2kc_distribution(path, ix, ix_good, ix_bad)
    # analysis_orn2pn.multiglo_lesion(path, ix, ix_good, ix_bad)

    default = {'kc_dropout_rate': 0.5, 'N_PN': 50}
    ykeys = ['val_acc', 'glo_score']
    xticks = [20, 50, 100, 200, 1000]
    sa.plot_results(path, xkey='N_PN', ykey=ykeys, loop_key='kc_dropout_rate',
                    logx=True, ax_args={'xticks': xticks}, figsize=(2.5, 1.5))
    select_dict = {'kc_dropout_rate': 0.5}
    sa.plot_progress(path, ykeys=ykeys, legend_key='N_PN',
                     select_dict=select_dict)


def control_vary_pn_relabel():
    config = FullConfig()
    config.data_dir = './datasets/proto/relabel_500_100'
    config.max_epoch = 100

    config.lr = 2e-4  # made smaller to improve separation
    config.initial_pn2kc = 4. / config.N_PN  # for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1. / config.N_PN

    config_ranges = OrderedDict()
    config_ranges['N_PN'] = [20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_pn_relabel_analysis(path):
    control_vary_pn_analysis(path)


#TODO
def controls_receptor():
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
    """Vary if initialization is dense or sparse"""
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
    """Vary APL."""
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


def pn_normalization_direct():
    '''
    Assesses the effect of PN normalization on glo score and performance
    '''
    config = FullConfig()
    config.skip_orn2pn = True
    config.train_pn2kc = False
    config.max_epoch = 30
    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['data_dir'] = ['./datasets/proto/concentration_mask',
                             # './datasets/proto/standard',
                             ]
    config_ranges['pn_norm_post'] = ['custom', 'None']

    # TODO: hyperparameter search
    # try:
    #     rmax = tools.load_pickles(path, 'model/layer1/r_max:0')
    #     print('rmax: {}'.format(rmax))
    #     rho = tools.load_pickles(path, 'model/layer1/rho:0')
    #     print('rho: {}'.format(rho))
    #     m = tools.load_pickles(path, 'model/layer1/m:0')
    #     print('m: {}'.format(m))
    # except:
    #     pass
    #
    # try:
    #     gamma = tools.load_pickles(path, 'model/layer1/LayerNorm/gamma:0')
    #     print('gamma params: {}'.format(gamma))
    # except:
    #     pass
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def train_claw_configs():
    '''
    Train (with or without loss) or fix connections from PN2KC while skipping ORN2PN
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7
    '''
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


def vary_kc_claws():
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
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
        sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss',
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
    # sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss', loop_key='ORN_NOISE_STD',
    #                 figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
    #                 ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})
    sa.plot_results(path, xkey='kc_inputs', ykey='val_logloss',
                    select_dict={'ORN_NOISE_STD': 0},
                    figsize=(2, 2),
                    ax_args={'ylim': [-1, 2], 'yticks': [-1, 0, 1, 2]})


def control_n_or_per_orn():
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

    config_ranges = OrderedDict()
    config_ranges['n_or_per_orn'] = list(range(1, 10))
    config_ranges['data_dir'] = ['./datasets/proto/n_or_per_orn'+str(n)
                             for n in range(1, 10)]
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_orn_corr():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.lr = 1e-4

    orn_corrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets = ['./datasets/proto/orn_corr_{:0.2f}'.format(c) for c in orn_corrs]

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
    ykeys = ['val_acc', 'K_inferred', 'glo_score']
    progress_keys = ['val_logloss', 'train_logloss', 'val_loss',
                     'train_loss', 'val_acc', 'glo_score', 'K_inferred']
    sa.plot_results(path, xkey=xkey, ykey=ykeys, figsize=(3.0, 1.5))
    sa.plot_progress(path, legend_key=xkey, ykeys=progress_keys)
    sa.plot_xy(path, xkey='lin_bins', ykey='lin_hist', legend_key=xkey,
               ax_args={'ylim': [0, 500]})
