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
import standard.experiment_controls
import standard.experiment_controls as experiment_controls
from standard.hyper_parameter_train import local_train, cluster_train
import matplotlib as mpl

SCRATCHPATH = '/axsys/scratch/ctn/projects/olfaction_evolution'
ROBERT_SCRATCHPATH = '/axsys/scratch/ctn/users/gy2259/olfaction_evolution'
PETER_SCRATCHPATH = '/axsys/scratch/ctn/users/yw2500/olfaction_evolution'

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-t', '--train', help='Training', action='store_true')
parser.add_argument('-a', '--analyze', help='Analyzing', action='store_true')
parser.add_argument('-test', '--testing', help='For debugging', action='store_true')
parser.add_argument('-e', '--experiment', nargs='+', help='Experiments', default='core')
parser.add_argument('-cp', '--clusterpath', help='cluster path', default=SCRATCHPATH)
parser.add_argument('-c', '--cluster', help='Use cluster?', action='store_true')
args = parser.parse_args()

# args.cluster_path = 'pw'
# args.cluster = True
# args.experiment = ['controls_glomeruli']
# args.train = True

print(args.__dict__)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
TRAIN, ANALYZE, is_test, use_cluster, cluster_path = args.train, args.analyze, args.testing, args.cluster, args.cluster_path

if use_cluster:
    train = cluster_train

    if cluster_path == 'peter' or cluster_path == 'pw':
        cluster_path = PETER_SCRATCHPATH
    elif cluster_path == 'robert' or cluster_path == 'gry':
        cluster_path = ROBERT_SCRATCHPATH
    else:
        cluster_path = SCRATCHPATH
else:
    train = local_train

if ANALYZE:
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

# experiments
if args.experiment == 'core':
    experiments = ['']
else:
    experiments = args.experiment


if 'controls_glomeruli' in experiments:
    # Vary ORN n duplication under different nKC
    path = './files/controls_glomeruli'
    if TRAIN:
        train(experiment_controls.controls_glomeruli(is_test), path, control=True)
    if ANALYZE:
        default = {'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5, 'N_ORN_DUPLICATION':10}

        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='glo_score',  figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='glo_score',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'N_ORN_DUPLICATION': 10, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='pn_norm_pre', y_key='glo_score',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'N_ORN_DUPLICATION': 10, 'ORN_NOISE_STD': 0, 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='kc_dropout_rate', y_key='glo_score',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'N_ORN_DUPLICATION':10})

        sa.plot_results(path, x_key='N_ORN_DUPLICATION', y_key='val_acc',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='ORN_NOISE_STD', y_key='val_acc',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'N_ORN_DUPLICATION': 10, 'pn_norm_pre': 'batch_norm', 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='pn_norm_pre', y_key='val_acc',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'N_ORN_DUPLICATION': 10, 'ORN_NOISE_STD': 0, 'kc_dropout_rate': 0.5})
        sa.plot_results(path, x_key='kc_dropout_rate', y_key='val_acc',figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                        select_dict={'ORN_NOISE_STD': 0, 'pn_norm_pre': 'batch_norm', 'N_ORN_DUPLICATION':10})