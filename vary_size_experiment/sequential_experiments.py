"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train


def vary_training(i):
    config = configs.FullConfig()
    config.save_path = './vary_KC_training/files/' + str(i).zfill(2)
    config.data_dir = '../datasets/proto/nodup'
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.max_epoch = 20

    config.skip_orn2pn = True
    config.direct_glo = False

    config.sign_constraint_pn2kc = True

    config.kc_inputs = 7
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['train_pn2kc'] = [True, True, False]
    hp_ranges['sparse_pn2kc'] = [False, False, True]
    hp_ranges['train_kc_bias'] = [False, False, True]
    hp_ranges['kc_loss'] = [False, True, False]
    hp_ranges['initial_pn2kc'] = [0.2, 0.2, 0]
    return config, hp_ranges

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for i in range(0,100):
        print('[***] Hyper-parameter: %2d' % i)
        tools.varying_config_sequential(vary_training, i)