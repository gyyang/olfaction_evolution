"""Vary noise level.

The goal of this set of experiments is to study the impact of noise
on the formation of glomeruli.
"""

import os
from collections import OrderedDict

import configs
import tools


def vary_kc(i):
    config = configs.FullConfig()
    config.save_path = './files/vary_noise3/' + str(i).zfill(2)
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.max_epoch = 30

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    return config, hp_ranges


def vary_n_orn_duplication(i):
    config = configs.FullConfig()
    config.save_path = './files/vary_n_orn_duplication/' + str(i).zfill(2)
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.max_epoch = 30
    config.N_KC = 100
    config.ORN_NOISE_STD = 0.5

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [1, 10]
    return config, hp_ranges


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # experiment = vary_kc
    experiment = vary_n_orn_duplication

    for i in range(0, 50):
        print('[***] Hyper-parameter: %2d' % i)
        tools.varying_config(experiment, i)