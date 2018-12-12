"""File that summarizes all key results.

To train and analyze all models quicly, run in command line
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
from standard.hyper_parameter_train import local_train, local_sequential_train
import standard.analysis as sa
import standard.analysis_pn2kc_training as pn2kc_training_analysis
import standard.analysis_pn2kc_random as pn2kc_random_analysis

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_true')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e','--experiment', nargs='+', help='Experiments', default='all')
args = parser.parse_args()

print(args)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN = args.train
ANALYZE = args.analyze
is_test = args.testing

# experiments
if args.experiment == 'all':
    experiments = ['orn2pn', 'vary_ORN_duplication', 'vary_PN', 'vary_KC',
                   'vary_KC_claws', 'train_KC_claws', 'random_KC_claws']
else:
    experiments = args.experiment

if 'orn2pn' in experiments:
    # Reproducing glomeruli-like activity
    path = './files/orn2pn'
    if TRAIN:
        local_train(se.train_orn2pn(is_test), path)
    if ANALYZE:
        sa.plot_progress(path)
        sa.plot_weights(path)

if 'vary_ORN_duplication' in experiments:
    # Vary ORN n duplication under different nKC
    path = './files/vary_ORN_duplication'
    if TRAIN:
        local_train(se.vary_orn_duplication_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_ORN_DUPLICATION',
                                       y_key='glo_score', loop_key='N_KC'),
        sa.plot_results(path, x_key='N_ORN_DUPLICATION',
                                       y_key='val_acc', loop_key='N_KC')

if 'vary_PN' in experiments:
    # Vary nPN under different noise levels
    path = './files/vary_PN'
    if TRAIN:
        local_train(se.vary_pn_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_PN', y_key='glo_score',
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_PN', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'vary_KC' in experiments:
    # Vary nKC under different noise levels
    path = './files/vary_KC'
    if TRAIN:
        local_train(se.vary_kc_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='N_KC', y_key='glo_score',
                                       loop_key='ORN_NOISE_STD'),
        sa.plot_results(path, x_key='N_KC', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'var_KC_claws' in experiments:
    path = './files/vary_KC_claws'
    if TRAIN:
        local_train(se.vary_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_results(path, x_key='kc_inputs', y_key='val_acc',
                                       loop_key='ORN_NOISE_STD')

if 'train_KC_claws' in experiments:
    path = './files/train_KC_claws'
    if TRAIN:
        local_sequential_train(se.train_claw_configs(is_test), path)
    if ANALYZE:
        sa.plot_progress(
            path, alpha=.75, linestyles=[':', '-.', '-'],
            legends=['Trainable, no loss', 'Trainable, with loss', 'Fixed']),
        pn2kc_training_analysis.plot_distribution(path)
        pn2kc_training_analysis.plot_sparsity(path)


if 'random_KC_claws' in experiments:
    path = './files/random_KC_claws'
    if TRAIN:
        local_train(se.random_claw_configs(is_test), path)
    if ANALYZE:
        pn2kc_random_analysis.plot_distribution(path)
        pn2kc_random_analysis.claw_distribution(path, 'random')
        pn2kc_random_analysis.plot_cosine_similarity(path, 'preserve', log=False)
        pn2kc_random_analysis.plot_cosine_similarity(path, 'random', log=False)
        pn2kc_random_analysis.pair_distribution(path, 'preserve')
        pn2kc_random_analysis.pair_distribution(path, 'random')