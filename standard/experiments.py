"""Experiments and corresponding analysis.

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

Analysis functions by convention are named
def name_analysis()
"""

from collections.__init__ import OrderedDict

import numpy as np

from configs import FullConfig, MetaConfig
from tools import vary_config
import tools

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


def standard():
    """Standard training setting"""
    config = FullConfig()
    config.max_epoch = 100

    config.pn_norm_pre = 'batch_norm'
    config.initial_pn2kc = 0.1
    config.save_every_epoch = True

    config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def standard_analysis(path):
    modeldirs = tools.get_modeldirs(path)
    dir = modeldirs[0]

    # accuracy
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_inferred'])

    # weight matrices
    sa.plot_weights(dir)

    try:
        analysis_orn2pn.correlation_across_epochs(path, arg='weight')
        analysis_orn2pn.correlation_across_epochs(path, arg='activity')
    except ModuleNotFoundError:
        pass

    # pn-kc
    analysis_pn2kc_training.plot_distribution(dir, xrange=1.5)
    analysis_pn2kc_training.plot_sparsity(dir, dynamic_thres=True, epoch=-1)

    # pn-kc random
    # analysis_pn2kc_random.plot_cosine_similarity(
    #     dir, shuffle_arg='preserve', log=False)
    # analysis_pn2kc_random.plot_distribution(dir)
    # analysis_pn2kc_random.claw_distribution(dir, shuffle_arg='random')
    # analysis_pn2kc_random.pair_distribution(dir, shuffle_arg='preserve')

    # Activity
    analysis_activity.distribution_activity(path, ['glo', 'kc'])
    analysis_activity.sparseness_activity(path, ['glo', 'kc'])


def receptor():
    """Standard training setting with full network including receptors."""
    config = FullConfig()
    config.max_epoch = 30

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.orn2pn_normalization = True
    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0.2

    # config.kc_norm_pre = 'batch_norm'
    # config.pn_norm_pre = 'batch_norm'

    config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    config_ranges['ORN_NOISE_STD'] = [0, 0.1, 0.2]
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def receptor_analysis(path):
    select_dict = dict()
    select_dict['kc_norm_pre'] = 'batch_norm'
    select_dict['ORN_NOISE_STD'] = 0.2
    select_dict['pn_norm_pre'] = None
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    dir = modeldirs[0]

    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_inferred'],
                     legend_key='pn_norm_pre')

    sa.plot_weights(dir)

    analysis_pn2kc_training.plot_distribution(dir, xrange=1.5)
    analysis_pn2kc_training.plot_sparsity(dir, dynamic_thres=True, epoch=-1)

    analysis_activity.distribution_activity(dir, ['glo', 'kc'])
    analysis_activity.sparseness_activity(dir, ['glo', 'kc'])


def standard_vary_hp():
    """Vary many hyperparameters for standard setting."""
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200

    config.pn_norm_pre = 'batch_norm'

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4]
    config_ranges['initial_pn2kc'] = np.array([2., 5., 10., 20.]) / config.N_PN

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def standard_vary_hp_analysis(path):
    select_dict = {}
    # select_dict['pn_norm_pre'] = 'batch_norm'
    exclude_dict = {}
    exclude_dict['kc_dropout_rate'] = 0.
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict,
                                    exclude_dict=exclude_dict, acc_min=0.8)
    print(len(modeldirs))
    # sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_inferred'])


def rnn():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.model = 'rnn'

    config.NEURONS = 2500
    config.BATCH_NORM = False

    config.dropout = True
    config.dropout_rate = 0
    config.DIAGONAL = True

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [1, 2, 3]
    config_ranges['replicate_orn_with_tiling'] = [False, True, True]
    config_ranges['N_ORN_DUPLICATION'] = [1, 10, 10]

    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def rnn_analysis(path):
    sa.plot_progress(path, ykeys=['val_acc'], legend_key='TIME_STEPS')
    # analysis_rnn.analyze_t0(path, dir_ix=0)
    analysis_rnn.analyze_t_greater(path, dir_ix=1)
    analysis_rnn.analyze_t_greater(path, dir_ix=2)


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
    # sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
    # sa.plot_weights(os.path.join(path, '0','epoch','2000'), var_name='w_glo', sort_axis=-1)
    analysis_pn2kc_training.plot_distribution(path, xrange=1)
    analysis_pn2kc_training.plot_sparsity(path, dir_ix=0, dynamic_thres=True,
                                          thres=.05)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)


def vary_pn():
    '''
    Vary number of PNs while fixing KCs to be 2500
    Results:
        GloScore should peak at PN=50, and then drop as PN > 50
        Accuracy should plateau at PN=50
        Results should be independent of noise
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    config_ranges = OrderedDict()
    config_ranges['N_PN'] = [10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_pn_analysis(path):
    xticks = [20, 50, 100, 200, 1000]
    ykeys = ['val_acc', 'glo_score']
    sa.plot_results(path, xkey='N_PN', ykey=ykeys, figsize=(1.75, 1.75),
                    loop_key='kc_dropout_rate', logx=True,
                    ax_args={'xticks': xticks})


def vary_kc():
    '''
    Vary number of KCs while also training ORN2PN.
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000, 20000]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_kc_analysis(path):
    xticks = [50, 200, 1000, 2500, 10000]
    ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
    ykeys = ['val_acc', 'glo_score']
    for ykey in ykeys:
        sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75),
                        ax_box=(0.25, 0.25, 0.65, 0.65),
                        loop_key='kc_dropout_rate', logx=True,
                        ax_args={'ylim': ylim, 'yticks': yticks,
                                 'xticks': xticks})


def vary_kc_activity_fixed():
    #TODO: use this one or the other one below
    '''

    :param argTest:
    :return:
    '''

    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    config.train_pn2kc = False

    # config.train_pn2kc = True
    # config.sparse_pn2kc = False
    # config.initial_pn2kc = .1
    # config.extra_layer = True
    # config.extra_layer_neurons = 200

    config_ranges = OrderedDict()
    config_ranges['kc_dropout_rate'] = [0, .5]
    x = [100, 200, 500, 1000, 2000, 5000]
    config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]
    configs = vary_config(config, config_ranges, mode='combinatorial')

    return configs


def vary_kc_activity_trainable():
    ''''''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # config.extra_layer = True
    # config.extra_layer_neurons = 200

    config_ranges = OrderedDict()
    config_ranges['kc_dropout_rate'] = [0, .5]
    x = [100, 200, 500, 1000, 2000, 5000]
    config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def pn_normalization():
    '''
    Assesses the effect of PN normalization on glo score and performance
    '''
    config = FullConfig()
    config.max_epoch = 15

    config.direct_glo = True
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1

    config.train_pn2kc = False

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    i = [0, .6]
    datasets = ['./datasets/proto/concentration_mask_row_' + str(s) for s in i]
    config_ranges['data_dir'] = ['./datasets/proto/standard'] + ['./datasets/proto/concentration'] + datasets
    config_ranges['pn_norm_pre'] = ['None','biology','fixed_activity']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def pn_normalization_analysis(path):
    sa.plot_results(path, xkey='data_dir', ykey='val_acc',
                    loop_key='pn_norm_pre',
                    select_dict={
                        'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                        'data_dir': ['./datasets/proto/standard',
                                     './datasets/proto/concentration_mask_row_0.6'
                                     ]},
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                    sort=False)

    sa.plot_results(path, xkey='data_dir', ykey='val_acc',
                    loop_key='pn_norm_pre',
                    select_dict={
                        'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                        'data_dir': ['./datasets/proto/concentration',
                                     './datasets/proto/concentration_mask_row_0',
                                     './datasets/proto/concentration_mask_row_0.6',
                                     ]},
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                    sort=False)
    # import tools
    # rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
    # rho = tools.load_pickle(path, 'model/layer1/rho:0')
    # m = tools.load_pickle(path, 'model/layer1/m:0')
    # print(rmax)
    # print(rho)
    # print(m)
    #
    # analysis_activity.image_activity(path, 'glo')
    # analysis_activity.image_activity(path, 'kc')
    # analysis_activity.distribution_activity(path, 'glo')
    # analysis_activity.distribution_activity(path, 'kc')
    # analysis_activity.sparseness_activity(path, 'kc')


def multihead():
    """Multi-task classification."""
    config = FullConfig()
    config.max_epoch = 30
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.pn_norm_pre = 'batch_norm'
    config.initial_pn2kc = 0.1

    config.save_every_epoch = False

    config.data_dir = './datasets/proto/multihead'

    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4]
    config_ranges['initial_pn2kc'] = [0.05, 0.1]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def multihead_analysis(path):
    modeldirs = tools.get_modeldirs(path, acc_min=0.75)
    analysis_multihead.analyze_many_networks_lesion(modeldirs)

    select_dict = dict()
    select_dict['pn_norm_pre'] = 'batch_norm'
    select_dict['lr'] = 1e-3
    select_dict['initial_pn2kc'] = 0.1
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict,
                                    acc_min=0.75)
    dir = modeldirs[0]
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score'])
    sa.plot_weights(dir)
    analysis_activity.distribution_activity(dir, ['glo', 'kc'])
    analysis_activity.sparseness_activity(dir, ['glo', 'kc'])
    analysis_multihead.analyze_example_network(dir)


def train_multihead_pruning():
    '''

    '''

    config = FullConfig()
    config.max_epoch = 30
    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    config.save_every_epoch = False
    config.initial_pn2kc = 10./config.N_PN
    config.kc_prune_threshold = 2./config.N_PN
    config.kc_prune_weak_weights = True

    config.data_dir = './datasets/proto/multi_head'

    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    config_ranges['dummy'] = [0, 1, 2]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs
