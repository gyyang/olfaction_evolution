import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

def no_pn_layer(i):
    config = configs.FullConfig()
    config.save_path = './no_pn_layer/test/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 1
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/_100_generalization_onehot'
    config.max_epoch = 5
    config.kc_norm_post = None
    config.train_pn2kc = True
    config.sign_constraint_pn2kc = True
    config.skip_orn2pn = True
    config.direct_glo = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['sparse_pn2kc'] = [True,False]
    return config, hp_ranges

def vary_dropout(i):
    config = configs.FullConfig()
    config.save_path = './dropout/normal_dup_0noise_range4/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 10
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.max_epoch = 5
    config.kc_norm_post = None
    config.train_pn2kc = False
    config.train_kc_bias = True
    config.sign_constraint_pn2kc = True
    config.skip_orn2pn = False
    config.uniform_pn2kc = False
    config.sparse_pn2kc = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_dropout'] = [True, False]
    return config, hp_ranges