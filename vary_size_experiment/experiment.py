"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

def vary_kc_claws(i):
    config = configs.FullConfig()
    config.save_path = './vary_KC_claws/uniform_bias_trainable/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/nodup'
    config.max_epoch = 20
    config.kc_norm_post = None
    config.train_pn2kc = False
    config.train_kc_bias = True
    config.sign_constraint_pn2kc = True
    config.skip_orn2pn = True
    config.direct_glo = False
    config.sparse_pn2kc = True
    config.uniform_pn2kc = True
    config.mean_subtract_pn2kc = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = list(range(1,15)) + \
                             list(range(15,30, 2)) + \
                             list(range(30, 50, 3))
    return config, hp_ranges

def temp(i):
    config = configs.FullConfig()
    config.save_path = './why_7_not_15/files/' + str(i).zfill(2)
    config.data_dir = '../datasets/proto/nodup'
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.max_epoch = 20

    config.skip_orn2pn = True
    config.direct_glo = False

    config.train_kc_bias = True
    config.kc_loss = False

    config.initial_pn2kc = 0
    config.initializer_pn2kc = 'normal'
    config.train_pn2kc = False
    config.sparse_pn2kc = True
    config.kc_inputs = 49
    config.mean_subtract_pn2kc = False
    config.save_every_epoch = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [3, 5, 7, 9, 11, 15, 20, 30, 40, 50]
    return config, hp_ranges

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(0,100):
        tools.varying_config(temp, i)
