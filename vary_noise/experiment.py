"""Vary noise level.

The goal of this set of experiments is to study the impact of noise
on the formation of glomeruli.
"""

import os
from collections import OrderedDict

import configs
import tools


def experiment(i):
    config = configs.FullConfig()
    config.save_path = './files/vary_noise2/' + str(i).zfill(2)
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.max_epoch = 30

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [100, 2500]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5]
    return config, hp_ranges


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for i in range(0, 100):
        print('[***] Hyper-parameter: %2d' % i)
        tools.varying_config(experiment, i)