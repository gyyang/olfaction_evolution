"""File that summarizes all key results.

To train and analyze all models quickly, run in command line
python paper.py -d=0 --train --analyze --testing

To reproduce the results from paper, run
python paper.py -d=0 --train --analyze

To analyze pretrained networks, run
python paper.py -d=0 --analyze

To run specific experiments (e.g. orn2pn, vary_pn), run
python paper.py -d=0 --train --analyze --experiment orn2pn vary_pn
"""

import os
import argparse

import standard.experiment as se
import standard.experiment_controls_pn2kc as experiments_controls_pn2kc
import standard.experiments_receptor as experiments_receptor
from standard.hyper_parameter_train import local_train, local_sequential_train, local_control_train
import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import standard.analysis_pn2kc_random as analysis_pn2kc_random
import standard.analysis_orn2pn as analysis_orn2pn
import standard.analysis_activity as analysis_activity
import standard.analysis_multihead as analysis_multihead
import standard.analysis_metalearn as analysis_metalearn
import oracle.evaluatewithnoise as evaluatewithnoise
import analytical.numerical_test as numerical_test
import analytical.analyze_simulation_results as analyze_simulation_results
import matplotlib as mpl

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_false')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e','--experiment', nargs='+', help='Experiments', default='core')
args = parser.parse_args()

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN = args.train
ANALYZE = args.analyze
is_test = args.testing

# experiments
if args.experiment == 'core':
    experiments = ['standard_without_or2orn', 'standard_with_or2orn',
                   'vary_orn_duplication', 'vary_pn',
                   'pn_normalization',
                   'vary_kc',
                   'vary_kc_activity_fixed', 'vary_kc_activity_trainable',
                   'vary_kc_claws', 'vary_kc_claws_new','train_kc_claws', 'random_kc_claws', 'train_orn2pn2kc',
                   'kc_claw_controls',
                   'or2orn', 'or2orn_duplication', 'or2orn_normalization',
                   'kcrole', 'kc_generalization',
                   'multi_head', 'metalearn']
else:
    experiments = args.experiment


#peter specific
#
TRAIN = True
ANALYZE = True
is_test = True
experiments = ['kc_claw_controls']


if 'standard_without_or2orn' in experiments:
    # Reproducing most basic findings
    path = './files/standard_net'
    if TRAIN:
        local_train(se.train_standardnet(is_test), path)
    if ANALYZE:
        # accuracy, glo score, cosine similarity
        sa.plot_progress(path, select_dict={'sign_constraint_orn2pn': True})
        analysis_pn2kc_random.plot_cosine_similarity(path, 'preserve', log=False)

        # #weights
        sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=0)
        sa.plot_weights(path, var_name='w_glo', dir_ix=0)

        # #sign constraint
        sa.plot_progress(path, legends=['Non-negative', 'No constraint'])
        sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key='glo_score')
        sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key='val_acc')

        #random analysis
        analysis_pn2kc_training.plot_distribution(path, xrange=1.5)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        analysis_pn2kc_random.plot_distribution(path)
        analysis_pn2kc_random.claw_distribution(path, 'random')
        analysis_pn2kc_random.pair_distribution(path, 'preserve')

        # # correlation
        analysis_orn2pn.get_correlation_coefficients(path, 'glo')
        sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key= 'glo_activity_corrcoef', yticks=[0, .25, .5],
                        ax_args={'ylim':[-.05, .5],'yticks':[0, .25, .5]})
        analysis_orn2pn.correlation_across_epochs(path, ['Non-negative', 'No constraint'])

        analysis_orn2pn.get_dimensionality(path, 'glo')
        sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key='glo_dimensionality')
        analysis_orn2pn.dimensionality_across_epochs(path, ['Non-negative', 'No constraint'])

if 'standard_with_or2orn' in experiments:
    path = './files/standard_net_with_or2orn'
    if TRAIN:
        local_train(se.train_standardnet_with_or2orn(is_test), path)
    if ANALYZE:
        # accuracy, glo score, cosine similarity
        # sa.plot_progress(path, select_dict={'sign_constraint_orn2pn': True})
        # analysis_pn2kc_random.plot_cosine_similarity(path, 'preserve', log=False)

        # #weights
        # sa.plot_weights(path, var_name='w_or', sort_axis=0, dir_ix=0)
        # sa.plot_weights(path, var_name='w_combined', dir_ix=0)
        # sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=0)
        sa.plot_weights(path, var_name='w_glo', dir_ix=0)

        # #sign constraint
        # sa.plot_progress(path, legends=['Non-negative', 'No constraint'])
        # sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key='glo_score')
        # sa.plot_results(path, x_key='sign_constraint_orn2pn', y_key='val_acc')
        #
        # #random analysis
        # analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=False)
        # analysis_pn2kc_training.plot_distribution(path, xrange=1.5, log=True)
        # analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        #
        # analysis_pn2kc_random.claw_distribution(path, 'random')
        # analysis_pn2kc_random.pair_distribution(path, 'preserve')

if 'vary_orn_duplication' in experiments:
    # Vary ORN n duplication under different nKC
    path = './files/vary_orn_duplication'
    if TRAIN:
        local_train(se.vary_orn_duplication_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_ORN_DUPLICATION',
                                       y_key='glo_score', loop_key='N_KC'),
        sa.plot_results(path, x_key='N_ORN_DUPLICATION',
                                       y_key='val_acc', loop_key='N_KC')

if 'vary_pn' in experiments:
    # Vary nPN under different noise levels
    path = './files/vary_pn'
    if TRAIN:
        local_train(se.vary_pn_configs(is_test), path)
    if ANALYZE:
        sa.plot_weights(path, sort_axis = 1, dir_ix=30, average=True)
        sa.plot_results(path, x_key='N_PN', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD':0}),
        sa.plot_results(path, x_key='N_PN', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        loop_key='ORN_NOISE_STD', plot_args= {'alpha':.75}
                        )

        sa.plot_results(path, x_key='N_PN', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        loop_key='ORN_NOISE_STD', plot_args= {'alpha':.75}
                        ),
        sa.plot_results(path, x_key='N_PN', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0})

        # # correlation and dimensionality
        # analysis_orn2pn.get_correlation_coefficients(path, 'glo')
        # sa.plot_results(path, x_key='N_PN', y_key= 'glo_activity_corrcoef', select_dict={'ORN_NOISE_STD':0},
        #                 yticks=[0, .25, .5],
        #                 ax_args={'ylim':[-.05, .5],'yticks':[0, .25, .5]})
        # analysis_orn2pn.get_dimensionality(path, 'glo')
        # sa.plot_results(path, x_key='N_PN', y_key= 'glo_dimensionality', select_dict={'ORN_NOISE_STD':0})


if 'vary_kc' in experiments:
    # Vary nKC under different noise levels
    path = './files/vary_kc'
    if TRAIN:
        local_train(se.vary_kc_configs(is_test), path)
    if ANALYZE:
        sa.plot_weights(path, sort_axis=1, dir_ix=0, average=True)
        sa.plot_results(path, x_key='N_KC', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0})
        sa.plot_results(path, x_key='N_KC', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0})
        sa.plot_results(path, x_key='N_KC', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_KC', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                                       loop_key='ORN_NOISE_STD')

        # # correlation and dimensionality
        # analysis_orn2pn.get_correlation_coefficients(path, 'glo')
        # sa.plot_results(path, x_key='N_KC', y_key= 'glo_activity_corrcoef', select_dict={'ORN_NOISE_STD':0},
        #                 yticks=[0, .1, .2],
        #                 ax_args={'ylim':[-.05, .2],'yticks':[0, .1, .2]})
        # analysis_orn2pn.get_dimensionality(path, 'glo')
        # sa.plot_results(path, x_key='N_KC', y_key= 'glo_dimensionality', select_dict={'ORN_NOISE_STD':0})

if 'vary_kc_no_dropout' in experiments:
    # Vary nKC under different noise levels
    path = './files/vary_kc_no_dropout'
    if TRAIN:
        local_train(se.vary_kc_no_dropout_configs(is_test), path)
    if ANALYZE:
        sa.plot_weights(path, sort_axis=1, dir_ix=0, average=True)
        sa.plot_results(path, x_key='N_KC', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0})
        sa.plot_results(path, x_key='N_KC', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0})
        sa.plot_results(path, x_key='N_KC', y_key='glo_score', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_KC', y_key='val_acc', figsize=(1.5, 1.5), ax_box = (0.27, 0.25, 0.65, 0.65),
                                       loop_key='ORN_NOISE_STD')

        # # correlation and dimensionality
        # analysis_orn2pn.get_correlation_coefficients(path, 'glo')
        # sa.plot_results(path, x_key='N_KC', y_key= 'glo_activity_corrcoef', select_dict={'ORN_NOISE_STD':0},
        #                 yticks=[0, .1, .2],
        #                 ax_args={'ylim':[-.05, .2],'yticks':[0, .1, .2]})
        # analysis_orn2pn.get_dimensionality(path, 'glo')
        # sa.plot_results(path, x_key='N_KC', y_key= 'glo_dimensionality', select_dict={'ORN_NOISE_STD':0})

if 'train_kc_claws' in experiments:
    path = './files/train_kc_claws'
    if TRAIN:
        local_sequential_train(se.train_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_progress(
            path, alpha=.75, linestyles=[':', '-'],
            legends=['Trained', 'Fixed']),
        sa.plot_weights(path, var_name='w_glo', sort_axis=-1, dir_ix=1)
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=False)
        analysis_pn2kc_training.plot_weight_distribution_per_kc(path, xrange=15)

if 'kc_claw_controls' in experiments:
    path = './files/kc_claw_controls'
    if TRAIN:
        local_control_train(experiments_controls_pn2kc.kc_claw_controls(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='kc_dropout_rate', dynamic_thres=True,
                                                      select_dict={'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm'})
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='ORN_NOISE_STD', dynamic_thres=True,
                                                      select_dict={'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5})
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='pn_norm_pre', dynamic_thres=True,
                                                      select_dict={'ORN_NOISE_STD': 0, 'kc_dropout_rate': 0.5})
        # analysis_pn2kc_training.plot_distribution(path)
        # analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        # default = {'ORN_NOISE_STD':0, 'pn_norm_pre':'batch_norm', 'kc_dropout_rate':0.5}
        # sa.plot_results(path, x_key='kc_dropout_rate', y_key='val_acc', ax_args ={'xticks': [0, .2, .4, .6]},
        #                 select_dict= {'ORN_NOISE_STD':0, 'pn_norm_pre':'batch_norm'})
        # sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='val_acc',
        #                 select_dict= {'pn_norm_pre':'batch_norm', 'kc_dropout_rate':0.5})
        # sa.plot_results(path, x_key='pn_norm_pre', y_key='val_acc',
        #                 select_dict= {'ORN_NOISE_STD':0, 'kc_dropout_rate':0.5})


if 'vary_kc_claws' in experiments:
    path = './files/vary_kc_claws'
    if TRAIN:
        local_train(se.vary_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc', loop_key='ORN_NOISE_STD',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc', select_dict={'ORN_NOISE_STD':0},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
        sa.plot_results(path, x_key='kc_inputs', y_key='val_loss', loop_key='ORN_NOISE_STD',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})
        sa.plot_results(path, x_key='kc_inputs', y_key='val_loss', select_dict={'ORN_NOISE_STD': 0},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        ax_args={'ylim':[-1, 2], 'yticks':[-1,0,1,2]})

if 'vary_kc_claws_new' in experiments:
    path = './files/vary_kc_claws_new'
    if TRAIN:
        local_train(se.vary_claw_configs_new(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc', loop_key='ORN_NOISE_STD',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc', select_dict={'ORN_NOISE_STD':0},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),)
        sa.plot_results(path, x_key='kc_inputs', y_key='val_loss', loop_key='ORN_NOISE_STD',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65))
        sa.plot_results(path, x_key='kc_inputs', y_key='val_loss', select_dict={'ORN_NOISE_STD': 0},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65))
        sa.plot_results(path, x_key='kc_inputs', y_key='train_loss',
                        select_dict={'ORN_NOISE_STD': 0},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65))

        evaluatewithnoise.evaluate_acrossmodels(path, select_dict={'ORN_NOISE_STD': 0})
        evaluatewithnoise.plot_acrossmodels(path)
        evaluatewithnoise.evaluate_acrossmodels(path, select_dict={
            'ORN_NOISE_STD': 0}, dataset='train')
        evaluatewithnoise.plot_acrossmodels(path, dataset='train')

if 'vary_kc_claws_dev' in experiments:
    path = './files/vary_kc_claws_epoch15'
    if TRAIN:
        local_train(se.vary_claw_configs_dev(is_test), path)
    if ANALYZE:
        evaluatewithnoise.evaluate_acrossmodels(
            path, select_dict={'ORN_NOISE_STD': 0},
            values=[0], n_rep=1, dataset='val', epoch=1)
        evaluatewithnoise.plot_acrossmodels(path, dataset='val', epoch=1)

if 'pn_normalization' in experiments:
    path = './files/pn_normalization'
    if TRAIN:
        local_train(se.pn_normalization(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='data_dir', y_key='val_acc', loop_key='pn_norm_pre',
                        select_dict={
                            'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                            'data_dir': ['./datasets/proto/standard',
                                         './datasets/proto/concentration_mask_row_0.6'
                                         ]},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)

        sa.plot_results(path, x_key='data_dir', y_key='val_acc', loop_key='pn_norm_pre',
                        select_dict={
                            'pn_norm_pre': ['None', 'fixed_activity', 'biology'],
                            'data_dir': ['./datasets/proto/concentration',
                                         './datasets/proto/concentration_mask_row_0',
                                         './datasets/proto/concentration_mask_row_0.6',
                                         ]},
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), sort=False)
        # import tools
        # rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
        # rho = tools.load_pickle(path, 'model/layer1/rho:0')
        # m = tools.load_pickle(path, 'model/layer1/m:0')
        # print(rmax)
        # print(rho)
        # print(m)
        #
        # analysis_activity.image_activity(path, 'glo_out')
        # analysis_activity.image_activity(path, 'kc_out')
        # analysis_activity.distribution_activity(path, 'glo_out')
        # analysis_activity.distribution_activity(path, 'kc_out')
        # analysis_activity.sparseness_activity(path, 'kc_out')

if 'or2orn' in experiments:
    path = './files/or2orn'
    if TRAIN:
        local_train(experiments_receptor.basic(is_test), path)
    if ANALYZE:
        sa.plot_progress(path)
        sa.plot_progress(path, select_dict={'ORN_NOISE_STD': 0.25})
        sa.plot_weights(path, var_name = 'w_or', sort_axis=0, dir_ix= 0)
        sa.plot_weights(path, var_name = 'w_or', sort_axis=0, dir_ix= 1)
        sa.plot_weights(path, var_name = 'w_orn', sort_axis= 1, dir_ix=0)
        sa.plot_weights(path, var_name = 'w_orn', sort_axis= 1, dir_ix=1)
        sa.plot_weights(path, var_name = 'w_combined', dir_ix=0)
        sa.plot_weights(path, var_name = 'w_combined', dir_ix=1)

if 'vary_or2orn_duplication' in experiments:
    path = './files/or2orn_orn_duplication'
    if TRAIN:
        local_train(experiments_receptor.vary_receptor_duplication(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='val_acc')
        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='glo_score')
        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='or_glo_score')
        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='combined_glo_score')

if 'vary_or2orn_normalization' in experiments:
    path = './files/or2orn_normalization'
    if TRAIN:
        local_train(experiments_receptor.vary_normalization(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='or2orn_normalization', y_key='val_acc', loop_key='orn2pn_normalization')
        sa.plot_results(path, x_key='or2orn_normalization', y_key='glo_score', loop_key='orn2pn_normalization')
        sa.plot_results(path, x_key='or2orn_normalization', y_key='or_glo_score', loop_key='orn2pn_normalization')
        sa.plot_results(path, x_key='or2orn_normalization', y_key='combined_glo_score', loop_key='orn2pn_normalization')
        
if 'multi_head' in experiments:
    path = './files/multi_head'
    if TRAIN:
        local_train(se.train_multihead(is_test), path)
    if ANALYZE:
        analysis_multihead.main1('multi_head')

if 'vary_kc_activity_fixed' in experiments:
    # Vary KC activity under different number of relabels
    path = './files/vary_kc_activity_fixed'
    if TRAIN:
        local_train(se.vary_kc_activity_fixed(is_test), path)
    if ANALYZE:
        # sa.plot_results(path, x_key='n_trueclass', y_key='val_acc', loop_key='kc_dropout_rate')
        analysis_activity.sparseness_activity(path, 'kc_out')
        analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')

if 'vary_kc_activity_trainable' in experiments:
    # Vary KC activity under different number of relabels
    path = './files/vary_kc_activity_trainable'
    if TRAIN:
        local_train(se.vary_kc_activity_trainable(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='n_trueclass', dynamic_thres=False, thres=.25)
        # sa.plot_results(path, x_key='n_trueclass', y_key='val_acc', loop_key='kc_dropout_rate')
        # analysis_activity.sparseness_activity(path, 'kc_out')
        # analysis_activity.plot_mean_activity_sparseness(path, 'kc_out', x_key='n_trueclass', loop_key='kc_dropout_rate')

if 'kcrole' in experiments:
    # Compare with or without KC layer
    path = './files/kcrole'
    if TRAIN:
        local_sequential_train(se.train_kcrole(is_test), path)
    if ANALYZE:
        # evaluatewithnoise.evaluate_kcrole(path, 'weight_perturb')
        evaluatewithnoise.plot_kcrole(path, 'weight_perturb')


if 'kc_generalization' in experiments:
    path = './files/kc_generalization'
    if TRAIN:
        local_sequential_train(se.kc_generalization(is_test), path)
    if ANALYZE:
        sa.plot_progress(path, legends=['No KC', 'Fixed KC'])

if 'metalearn' in experiments:
    path = './files/metalearn'
    if TRAIN:
        local_sequential_train(se.metalearn(is_test), path, train_arg='metalearn')
    if ANALYZE:
        # analysis_pn2kc_training.plot_distribution(path, xrange=1)
        # analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True, thres=.05)
        # sa.plot_weights(path, var_name='w_glo', sort_axis=-1, dir_ix=1)
        # sa.plot_weights(path, var_name='w_orn', sort_axis=1, dir_ix=-0, average=False)
        analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_orn', dir_ix = 0)
        analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'w_glo', dir_ix= 1)
        analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 0)
        analysis_metalearn.plot_weight_change_vs_meta_update_magnitude(path, 'model/layer3/kernel:0', dir_ix = 1)

if 'vary_n_orn' in experiments:
    # Train networks with different numbers of ORs
    path = './files/vary_n_orn2'
    if TRAIN:
        import paper_datasets
        paper_datasets.make_vary_or_datasets()
        local_sequential_train(se.vary_n_orn(is_test), path)
    if ANALYZE:
        pass

if 'frequent_eval' in experiments:
    path = './files/frequent_eval'
    if TRAIN:
        local_train(se.vary_claw_configs_frequentevaluation(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), )
        sa.plot_results(path, x_key='kc_inputs', y_key='val_loss',
                        figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65), )

if 'analytical' in experiments:
    if TRAIN:
        numerical_test.get_optimal_K_simulation()
    if ANALYZE:
        numerical_test.main_compare()
        numerical_test.main_plot()
        analyze_simulation_results.main()

if 'apl' in experiments:
    path = './files/apl'
    if TRAIN:
        local_train(se.vary_apl(is_test), path)
    if ANALYZE:
        analysis_activity.sparseness_activity(
            path, 'kc_out', activity_threshold=0., lesion_kwargs=None)
        lk = {'name': 'model/apl2kc/kernel:0',
              'units': 0, 'arg': 'outbound'}
        analysis_activity.sparseness_activity(
            path, 'kc_out', activity_threshold=0., lesion_kwargs=lk,
            figname='lesion_apl_')

if 'meansub' in experiments:
    path = './files/meansub'
    if TRAIN:
        local_sequential_train(se.vary_w_glo_meansub_coeff(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.plot_pn2kc_claw_stats(
            path, x_key='w_glo_meansub_coeff', dynamic_thres=True)