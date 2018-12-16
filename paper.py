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
from standard.hyper_parameter_train import local_train, local_sequential_train
import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import standard.analysis_pn2kc_random as analysis_pn2kc_random

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_false')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e','--experiment', nargs='+', help='Experiments', default='core')
args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN = args.train
ANALYZE = args.analyze
is_test = args.testing

# experiments
if args.experiment == 'core':
    experiments = ['orn2pn', 'vary_orn_duplication', 'vary_pn', 'vary_kc',
                   'vary_kc_claws', 'train_kc_claws', 'random_kc_claws', 'train_orn2pn2kc',
                   'vary_pn2kc_loss', 'vary_pn2kc_initial_value','vary_pn2kc_noise']
else:
    experiments = args.experiment

# #peter specific
# TRAIN = True
# ANALYZE = True
# is_test = True
# experiments = ['vary_kc_claws']

if 'orn2pn' in experiments:
    # Reproducing glomeruli-like activity
    path = './files/orn2pn'
    if TRAIN:
        local_train(se.train_orn2pn(is_test), path)
    if ANALYZE:
        sa.plot_progress(path)
        sa.plot_weights(path)

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
        sa.plot_results(path, x_key='N_PN', y_key='glo_score',
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_PN', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'vary_kc' in experiments:
    # Vary nKC under different noise levels
    path = './files/vary_kc'
    if TRAIN:
        local_train(se.vary_kc_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_KC', y_key='glo_score',
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_KC', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'vary_kc_claws' in experiments:
    path = './files/vary_kc_claws'
    if TRAIN:
        local_train(se.vary_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'train_kc_claws' in experiments:
    path = './files/train_kc_claws'
    if TRAIN:
        local_sequential_train(se.train_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_progress(
            path, alpha=.75, linestyles=[':', '-.', '-'],
            legends=['Trainable, no loss', 'Trainable, with loss', 'Fixed']),
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path)


if 'random_kc_claws' in experiments:
    path = './files/random_kc_claws'
    if TRAIN:
        local_train(se.random_claw_configs(is_test), path)
    if ANALYZE:
        analysis_pn2kc_random.plot_distribution(path)
        analysis_pn2kc_random.claw_distribution(path, 'random')
        analysis_pn2kc_random.plot_cosine_similarity(path, 'preserve', log=False)
        analysis_pn2kc_random.plot_cosine_similarity(path, 'random', log=False)
        analysis_pn2kc_random.pair_distribution(path, 'preserve')
        analysis_pn2kc_random.pair_distribution(path, 'random')

if 'vary_norm' in experiments:
    path = './files/vary_norm'
    if TRAIN:
        local_train(se.vary_norm(is_test), path)
    if ANALYZE:
        pass

if 'train_orn2pn2kc' in experiments:
    path = './files/train_orn2pn2kc'
    if TRAIN:
        local_train(se.train_orn2pn2kc_configs(is_test), path)
    if ANALYZE:
        sa.plot_progress(
            path, alpha=.75, linestyles=[':', '-.', '-'],
            legends=['No Noise', ' 0.5 Noise', '1.0 Noise']),
        sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='glo_score')
        sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='val_acc')
        sa.plot_weights(path)
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path)

if 'vary_pn2kc_initial_value' in experiments:
    path = './files/vary_pn2kc_initial_value'
    if TRAIN:
        local_train(experiments_controls_pn2kc.vary_pn2kc_initial_value_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='initial_pn2kc', y_key='val_acc')
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path)
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='initial_pn2kc')
        analysis_pn2kc_training.plot_weight_distribution_per_kc(path)

if 'vary_pn2kc_loss' in experiments:
    path = './files/vary_pn2kc_loss'
    if TRAIN:
        local_train(experiments_controls_pn2kc.vary_pn2kc_loss_configs(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.image_pn2kc_parameters(path)
        # pn2kc_training_analysis.plot_distribution(path)
        # sa.plot_results(path, x_key='kc_loss_beta', y_key='glo_score', loop_key='kc_loss_alpha')
        # sa.plot_results(path, x_key='kc_loss_beta', y_key='val_acc', loop_key='kc_loss_alpha')

if 'vary_pn2kc_noise' in experiments:
    path = './files/vary_pn2kc_noise'
    if TRAIN:
        local_train(experiments_controls_pn2kc.vary_pn2kc_noise_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='val_acc', loop_key= 'N_ORN_DUPLICATION')
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='ORN_NOISE_STD', loop_key='N_ORN_DUPLICATION')

        # pn2kc_training_analysis.plot_distribution(path)
        # sa.plot_results(path, x_key='kc_loss_beta', y_key='glo_score', loop_key='kc_loss_alpha')
        # sa.plot_results(path, x_key='kc_loss_beta', y_key='val_acc', loop_key='kc_loss_alpha')
