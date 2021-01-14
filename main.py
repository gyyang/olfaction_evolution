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
