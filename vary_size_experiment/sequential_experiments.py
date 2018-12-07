"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train


def kc_weight_changes(i):
    config = configs.FullConfig()
    config.save_path = './kc_weight_change/files/' + str(i).zfill(2)
    config.data_dir = '../datasets/proto/nodup'
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.max_epoch = 10

    config.skip_orn2pn = True
    config.direct_glo = False

    config.kc_norm_post = None
    config.train_kc_bias = True
    config.sign_constraint_pn2kc = True
    config.mean_subtract_pn2kc = False

    config.kc_inputs = 7
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['uniform_pn2kc'] = [False, True, False]
    hp_ranges['train_pn2kc'] = [True, False, True]
    hp_ranges['sparse_pn2kc'] = [False, True, False]
    return config, hp_ranges

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for i in range(2,100):
        print('[***] Hyper-parameter: %2d' % i)
        tools.varying_config_sequential(kc_weight_changes, i)