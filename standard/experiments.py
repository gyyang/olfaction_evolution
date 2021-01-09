"""Experiments and corresponding analysis.

Each experiment is described by a function that returns a list of configurations
function name is the experiment name

Analysis functions by convention are named
def name_analysis()
"""

from collections.__init__ import OrderedDict

import numpy as np

from configs import FullConfig, MetaConfig, RNNConfig, SingleLayerConfig
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


def standard():
    """Standard training setting"""
    config = FullConfig()
    # config.max_epoch = 100

    # config.pn_norm_pre = 'batch_norm'
    # config.initial_pn2kc = 0.1
    config.save_every_epoch = True

    # config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def standard_analysis(path):
    modeldirs = tools.get_modeldirs(path)
    dir = modeldirs[0]

    # accuracy
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_smart'])

    # weight matrices
    sa.plot_weights(dir)

    if not use_torch:
        # This currently doesn't work for pytorch
        try:
            analysis_orn2pn.correlation_across_epochs(path, arg='weight')
            analysis_orn2pn.correlation_across_epochs(path, arg='activity')
        except ModuleNotFoundError:
            pass

    # pn-kc
    analysis_pn2kc_training.plot_distribution(dir, xrange=1.5)
    analysis_pn2kc_training.plot_sparsity(dir, dynamic_thres=True, epoch=-1)

    analysis_pn2kc_training.plot_log_distribution_movie(dir)

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
    # config.max_epoch = 200

    config.receptor_layer = True
    # config.or2orn_normalization = True
    # config.orn2pn_normalization = True
    # config.replicate_orn_with_tiling = True
    # config.N_ORN_DUPLICATION = 10
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
    select_dict = {'ORN_NOISE_STD': 0.1, 'kc_norm_pre': 'batch_norm',
                   'pn_norm_pre': None}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    dir = modeldirs[0]
    # accuracy
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_smart'])

    # weight matrices
    sa.plot_weights(dir)
    sa.plot_weights(dir, zoomin=True)

    # pn-kc
    analysis_pn2kc_training.plot_distribution(dir, xrange=1.5)
    analysis_pn2kc_training.plot_sparsity(dir, dynamic_thres=True, epoch=-1)
    analysis_pn2kc_training.plot_log_distribution_movie(dir)

    # Compute glo-score metric for OR-ORN connectivity
    print('ORN score for OR-ORN connectivity',
          tools.compute_glo_score(tools.load_pickle(dir)['w_or'], 50)[0])


def singlelayer():
    """Standard single layer training setting"""
    config = SingleLayerConfig()
    config.max_epoch = 30

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def receptor_multilr():
    """Standard training setting with full network including receptors."""
    config = FullConfig()
    config.max_epoch = 1000

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.orn2pn_normalization = True
    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0.2

    config.kc_norm_pre = 'batch_norm'
    config.pn_norm_pre = None

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10./config.N_PN
    config.lr = 2e-3

    config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    # config_ranges['ORN_NOISE_STD'] = [0.2]
    # config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    # config_ranges['kc_norm_pre'] = [None, 'batch_norm']
    config_ranges['lr'] = [2e-3, 2e-4]
    config_ranges['pn2kc_lr'] = [2e-3, 2e-4]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def receptor_multilr_analysis(path):
    select_dict = dict()
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict, acc_min=0.5)
    sa.plot_progress(modeldirs, ykeys=['val_acc', 'glo_score', 'K_smart'])

    # select_dict = dict()
    # select_dict['kc_norm_pre'] = 'batch_norm'
    # select_dict['ORN_NOISE_STD'] = 0.2
    # select_dict['pn_norm_pre'] = None
    # modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    # modeldir = modeldirs[0]
    select_dict = {}
    modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
    modeldirs = analysis_pn2kc_training.filter_modeldirs(
        modeldirs, exclude_badkc=True, exclude_badpeak=True)

    for modeldir in modeldirs:
        # sa.plot_weights(modeldir)

        analysis_pn2kc_training.plot_distribution(modeldir)
        analysis_pn2kc_training.plot_sparsity(modeldir, epoch=-1)

        # analysis_pn2kc_training.plot_log_distribution_movie(modeldir)

        # analysis_activity.distribution_activity(modeldir, ['glo', 'kc'])
        # analysis_activity.sparseness_activity(modeldir, ['glo', 'kc'])


def rnn():
    config = RNNConfig()
    config.max_epoch = 100
    config.rec_dropout = True
    config.rec_dropout_rate = 0.5
    config.rec_norm_pre = None
    config.diagonal = True
    config.ORN_NOISE_STD = 0.0

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [2]
    config_ranges['rec_dropout_rate'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def rnn_wdropout():
    # TEMPORARY
    config = RNNConfig()
    config.max_epoch = 30
    config.rec_dropout = False
    config.weight_dropout = True
    config.rec_norm_pre = None
    config.diagonal = True
    config.ORN_NOISE_STD = 0.0

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [2]
    config_ranges['weight_dropout_rate'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def rnn_tf():
    # TODO: To be removed in the future
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
    # sa.plot_progress(path, ykeys=['val_acc'], legend_key='TIME_STEPS')
    # analysis_rnn.analyze_t0(path, dir_ix=0)
    analysis_rnn.analyze_t_greater(path, dir_ix=1)
    # analysis_rnn.analyze_t_greater(path, dir_ix=2)


def rnn_relabel():
    config = RNNConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.rec_dropout = True
    config.rec_dropout_rate = 0.0
    config.rec_norm_pre = None
    config.diagonal = True
    config.ORN_NOISE_STD = 0.0

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [2]
    config_ranges['diagonal'] = [True, False]
    config_ranges['lr'] = [1e-3, 5e-4, 2e-4, 1e-4]
    # config_ranges['rec_dropout_rate'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def rnn_relabel_prune():
    config = RNNConfig()
    config.data_dir = './datasets/proto/relabel_200_100'
    config.max_epoch = 100
    config.rec_dropout = False
    config.rec_dropout_rate = 0.0
    config.rec_norm_pre = None
    config.diagonal = False
    config.ORN_NOISE_STD = 0.0

    config.prune_weak_weights = True
    config.prune_threshold = 1. / config.NEURONS
    config.initial_rec = 4./config.NEURONS

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [1, 2, 3]
    config_ranges['lr'] = [1e-3, 5e-4, 2e-4, 1e-4]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def rnn_relabel_analysis(path):
    select_dict = {'diagonal': True}
    sa.plot_progress(path, ykeys=['val_acc'], legend_key='lr', select_dict=select_dict)
    sa.plot_results(path, xkey='lr', ykey='val_acc', select_dict=select_dict)


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
    analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True,
                                          thres=.05)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
    # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)


# def vary_pn():
#     '''
#     Vary number of PNs while fixing KCs to be 2500
#     Results:
#         GloScore should peak at PN=50, and then drop as PN > 50
#         Accuracy should plateau at PN=50
#         Results should be independent of noise
#     '''
#     config = FullConfig()
#     config.data_dir = './datasets/proto/standard'
#     config.max_epoch = 30
#     config.pn_norm_pre = 'batch_norm'
#
#     config_ranges = OrderedDict()
#     config_ranges['N_PN'] = [10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
#
#     configs = vary_config(config, config_ranges, mode='combinatorial')
#     return configs
#
#
# def vary_pn_analysis(path):
#     xticks = [20, 50, 100, 200, 1000]
#     ykeys = ['val_acc', 'glo_score']
#     sa.plot_results(path, xkey='N_PN', ykey=ykeys, figsize=(1.75, 1.75),
#                     loop_key='kc_dropout_rate', logx=True,
#                     ax_args={'xticks': xticks})
#
#
# def vary_kc():
#     '''
#     Vary number of KCs while also training ORN2PN.
#     '''
#     config = FullConfig()
#     config.data_dir = './datasets/proto/standard'
#     config.max_epoch = 30
#     config.pn_norm_pre = 'batch_norm'
#
#     # Ranges of hyperparameters to loop over
#     config_ranges = OrderedDict()
#     config_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000, 20000]
#
#     configs = vary_config(config, config_ranges, mode='combinatorial')
#     return configs
#
#
# def vary_kc_analysis(path):
#     xticks = [50, 200, 1000, 2500, 10000]
#     ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
#     ykeys = ['val_acc', 'glo_score']
#     for ykey in ykeys:
#         sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75),
#                         ax_box=(0.25, 0.25, 0.65, 0.65),
#                         loop_key='kc_dropout_rate', logx=True,
#                         ax_args={'ylim': ylim, 'yticks': yticks,
#                                  'xticks': xticks})


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
    # rmax = tools.load_pickles(path, 'model/layer1/r_max:0')
    # rho = tools.load_pickles(path, 'model/layer1/rho:0')
    # m = tools.load_pickles(path, 'model/layer1/m:0')
    # print(rmax)
    # print(rho)
    # print(m)
    #
    # analysis_activity.image_activity(path, 'glo')
    # analysis_activity.image_activity(path, 'kc')
    # analysis_activity.distribution_activity(path, 'glo')
    # analysis_activity.distribution_activity(path, 'kc')
    # analysis_activity.sparseness_activity(path, 'kc')


def vary_or_prune(n_pn=50):
    """Training networks with different number of PNs and vary hyperparams."""
    config = FullConfig()
    config.max_epoch = 100
    config.save_log_only = True

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True  # Skip ORN-to-PN

    config.initial_pn2kc = 4. / config.N_PN  # explicitly set for clarity
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./n_pn

    config.N_KC = min(40000, n_pn**2)

    config_ranges = OrderedDict()
    config_ranges['lr'] = [1e-3, 5e-4, 2e-4, 1e-4]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_or_prune_analysis(path, n_pn=None):
    def _vary_or_prune_analysis(path, n_pn):
        # Analyze individual network
        select_dict = {}
        modeldirs = tools.get_modeldirs(path, select_dict=select_dict)
        _modeldirs = modeldirs
        # _modeldirs = analysis_pn2kc_training.filter_modeldirs(
        #     modeldirs, exclude_badkc=True, exclude_badpeak=True)
        sa.plot_progress(_modeldirs, ykeys=['val_acc', 'K_smart'],
                         legend_key='lr')

        _modeldirs = modeldirs
        sa.plot_xy(_modeldirs,
                   xkey='lin_bins', ykey='lin_hist', legend_key='lr',
                   ax_args={'ylim': [0, n_pn ** 2.4 / 50],
                            'xlim': [0, 8 / n_pn**0.6]})
        # x, y lim from heuristics, exponent should sum to 2-3

    if n_pn is not None:
        _vary_or_prune_analysis(path, n_pn)

    else:
        import glob
        path = path + '_pn'  # folders named XX_pn50, XX_pn100, ..
        folders = glob.glob(path + '*')
        n_orns = sorted([int(folder.split(path)[-1]) for folder in folders])
        Ks = list()
        # n_orns = [25, 75, 100, 150, 200]
        for n_orn in n_orns:
            _path = path + str(n_orn)
            _vary_or_prune_analysis(_path, n_pn=n_orn)
            modeldirs = tools.get_modeldirs(_path, acc_min=0.75)
            modeldirs = analysis_pn2kc_training.filter_modeldirs(
                modeldirs, exclude_badkc=True, exclude_badpeak=True)

            # Use model with highest LR among good models
            modeldirs = tools.sort_modeldirs(modeldirs, 'lr')
            modeldirs = [modeldirs[-1]]

            res = tools.load_all_results(modeldirs)
            Ks.append(res['K_smart'])

        for plot_dim in [False, True]:
            analysis_pn2kc_training.plot_all_K(n_orns, Ks, plot_box=True,
                                               plot_dim=plot_dim,
                                               path='vary_or_prune')


def control_pn2kc_prune_hyper_analysis(path, n_pns):
    import copy
    for n_pn in n_pns:
        cur_path = path + '_' + str(n_pn)
        default = {'N_KC': 2500, 'lr': 1e-3, 'initial_pn2kc': 4. / n_pn,
                   'kc_prune_threshold': 1. / n_pn}
        ykeys = ['val_acc', 'K']
        for yk in ykeys:
            exclude_dict = None
            if yk in ['K_smart', 'sparsity_inferred', 'K', 'sparsity']:
                # exclude_dict = {'lr': [3e-3, 1e-2, 3e-2]}
                pass

            for xk, v in default.items():
                temp = copy.deepcopy(default)
                temp.pop(xk)
                logx = True
                # sa.plot_results(cur_path, xkey=k, ykey=yk, figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                #                 select_dict=temp,
                #                 logx=logx)
                #
                # sa.plot_progress(cur_path, select_dict=temp, ykeys=[yk], legend_key=k, exclude_dict=exclude_dict)
        #
        res = standard.analysis_pn2kc_peter.do_everything(cur_path,
                                                          filter_peaks=True,
                                                          redo=True, range=.75)
        for xk, v in default.items():
            temp = copy.deepcopy(default)
            temp.pop(xk)
            sa.plot_xy(cur_path, select_dict=temp, xkey='lin_bins_',
                       ykey='lin_hist_', legend_key=xk, log=res,
                       ax_args={'ylim': [0, 500]})


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
    # this acc is average of two heads
    modeldirs = tools.get_modeldirs(path, acc_min=0.85)
    modeldirs = analysis_pn2kc_training.filter_modeldirs(
        modeldirs, exclude_badkc=True)
    analysis_multihead.analyze_many_networks_lesion(modeldirs)

    select_dict = {}
    select_dict['lr'] = 1e-3
    select_dict['pn_norm_pre'] = 'batch_norm'
    modeldirs = tools.get_modeldirs(path, acc_min=0.85, select_dict=select_dict)
    modeldirs = analysis_pn2kc_training.filter_modeldirs(
        modeldirs, exclude_badpeak=True)
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
