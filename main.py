"""File that summarizes all key results.

To train specific experiments (e.g. orn2pn, vary_pn), run
python main.py --train experiment_name

To analyze specific experiments (e.g. orn2pn, vary_pn), run
python main.py --analyze experiment_name

To train models quickly, run in command line
python main.py --train experiment_name --testing
"""

import platform
import os
import argparse

from standard.experiment_utils import train_experiment, analyze_experiment
from paper_datasets import make_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', nargs='+', help='Train experiments', default=[])
parser.add_argument('-a', '--analyze', nargs='+', help='Analyze experiments', default=[])
parser.add_argument('-data', '--dataset', nargs='+', help='Make datasets', default=[])
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-n', '--n_pn', help='Number of olfactory receptors', default=None, type=int)
args = parser.parse_args()

for item in args.__dict__.items():
    print(item)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

experiments2train = args.train
experiments2analyze = args.analyze
datasets = args.dataset
testing = args.testing
n_pn = args.n_pn
use_cluster = 'columbia' in platform.node()  # on columbia cluster

if 'core' in experiments2train:
    experiments2train = [
        'standard',
        'receptor',
        'vary_pn',
        'vary_kc',
        'metalearn',
        'pn_normalization',
        'vary_kc_activity_fixed', 'vary_kc_activity_trainable',
        'vary_kc_claws', 'vary_kc_claws_new','train_kc_claws',
        'random_kc_claws', 'train_orn2pn2kc',
        'kcrole', 'kc_generalization',
        'multi_head']

if 'supplement' in experiments2train:
    experiments2train = []  # To be added

for experiment in experiments2train:
    train_experiment(experiment, use_cluster=use_cluster, testing=testing,
                     n_pn=n_pn)

for experiment in experiments2analyze:
    analyze_experiment(experiment, n_pn=n_pn)

for dataset in datasets:
    make_dataset(dataset)


# if 'standard' in experiments:
#     path = './files/standard'
#     if ANALYZE:
#         # accuracy
#         sa.plot_progress(path, ykeys=['val_acc', 'glo_score', 'K_inferred'])
#
#         # orn-pn
#         sa.plot_weights(os.path.join(path,'000000'), var_name='w_orn', sort_axis=1)
#         try:
#             analysis_orn2pn.correlation_across_epochs(path, arg='weight')
#             analysis_orn2pn.correlation_across_epochs(path, arg='activity')
#         except ModuleNotFoundError:
#             pass
#
#         # pn-kc
#         sa.plot_weights(os.path.join(path,'000000'), var_name='w_glo')
#
#         # pn-kc
#         analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=False)
#         analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=True)
#         analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, epoch=-1)
#
#         # pn-kc random
#         analysis_pn2kc_random.plot_cosine_similarity(path, dir_ix= 0, shuffle_arg='preserve', log=False)
#         analysis_pn2kc_random.plot_distribution(path, dir_ix= 0)
#         analysis_pn2kc_random.claw_distribution(path, dir_ix= 0, shuffle_arg='random')
#         analysis_pn2kc_random.pair_distribution(path, dir_ix= 0, shuffle_arg='preserve')

# if 'receptor' in experiments:
#     path = './files/receptor'
#     if ANALYZE:
#         sa.plot_progress(path, ykeys=['val_acc', 'glo_score', 'K_inferred'])
#
#         for var_name in ['w_or', 'w_orn', 'w_combined', 'w_glo']:
#             sort_axis = 0 if var_name == 'w_or' else 1
#             sa.plot_weights(os.path.join(path, '000000'),
#                             var_name=var_name, sort_axis=sort_axis)
#
#         # pn-kc K
#         analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=False)
#         analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=True)
#         analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, epoch=-1)

# if 'vary_pn' in experiments:
#     # Vary nPN
#     path = './files/vary_pn'
#     if ANALYZE:
#         xticks = [20, 50, 100, 200, 1000]
#         ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
#         ykeys = ['val_acc', 'glo_score']
#         for ykey in ykeys:
#             sa.plot_results(path, xkey='N_PN', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
#                             loop_key='kc_dropout_rate', logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks':xticks})

# if 'vary_kc' in experiments:
#     # Vary nKC
#     path = './files/vary_kc'
#     if ANALYZE:
#         xticks = [50, 200, 1000, 2500, 10000]
#         ylim, yticks = [0, 1.05], [0, .25, .5, .75, 1]
#         ykeys = ['val_acc', 'glo_score']
#         for ykey in ykeys:
#             sa.plot_results(path, xkey='N_KC', ykey=ykey, figsize=(1.75, 1.75), ax_box=(0.25, 0.25, 0.65, 0.65),
#                             loop_key='kc_dropout_rate', logx=True, ax_args={'ylim': ylim, 'yticks': yticks, 'xticks':xticks})

# if 'rnn' in experiments:
#     path = './files/rnn'
#     if ANALYZE:
#         sa.plot_progress(path, ykeys=['val_acc'], legend_key='TIME_STEPS')
#         # analysis_rnn.analyze_t0(path, dir_ix=0)
#         analysis_rnn.analyze_t_greater(path, dir_ix=1)
#         analysis_rnn.analyze_t_greater(path, dir_ix=2)

# if 'metalearn' in experiments:
#     path = './files/metalearn'
#     if ANALYZE:
#         # sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
#         # sa.plot_weights(os.path.join(path, '0','epoch','2000'), var_name='w_glo', sort_axis=-1)
#         analysis_pn2kc_training.plot_distribution(path, xrange=1)
#         analysis_pn2kc_training.plot_sparsity(path, dir_ix=0, dynamic_thres=True, thres=.05)
#         # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
#         # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
#         # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
#         # analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)

# if 'pn_normalization' in experiments:
#     path = './files/pn_normalization'
#     if ANALYZE:
#         sa.plot_results(path, xkey='data_dir', ykey='val_acc', loop_key='pn_norm_pre',
#                         select_dict={
#                             'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
#                             'data_dir': ['./datasets/proto/standard',
#                                          './datasets/proto/concentration_mask_row_0.6'
#                                          ]},
#                         figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)
#
#         sa.plot_results(path, xkey='data_dir', ykey='val_acc', loop_key='pn_norm_pre',
#                         select_dict={
#                             'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
#                             'data_dir': ['./datasets/proto/concentration',
#                                          './datasets/proto/concentration_mask_row_0',
#                                          './datasets/proto/concentration_mask_row_0.6',
#                                          ]},
#                         figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)
#         # import tools
#         # rmax = tools.load_pickles(path, 'model/layer1/r_max:0')
#         # rho = tools.load_pickles(path, 'model/layer1/rho:0')
#         # m = tools.load_pickles(path, 'model/layer1/m:0')
#         # print(rmax)
#         # print(rho)
#         # print(m)
#         #
#         # analysis_activity.image_activity(path, 'glo_out')
#         # analysis_activity.image_activity(path, 'kc_out')
#         # analysis_activity.distribution_activity(path, 'glo_out')
#         # analysis_activity.distribution_activity(path, 'kc_out')
#         # analysis_activity.sparseness_activity(path, 'kc_out')
        
# if 'multi_head' in experiments:
#     path = './files/multi_head'
#     if ANALYZE:
#         # analysis_multihead.main1('multi_head')
#         sa.plot_weights(os.path.join(path, '000000'), var_name='w_orn', sort_axis=1)
