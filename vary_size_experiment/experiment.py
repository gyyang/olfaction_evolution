"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

def vary_pn(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.N_ORN = 50
    config.N_ORN_DUPLICATION = 10
    config.max_epoch = 5
    config.save_path = './vary_PN/files/' + str(i).zfill(2)

    hp_ranges = OrderedDict()
    hp_ranges['N_PN'] = [10,30,50,100]
    # hp_ranges['N_PN'] = [200]
    return config, hp_ranges

def vary_kc(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot'
    config.N_ORN = 50
    config.N_PN = 50
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.train_kc_bias = True
    config.max_epoch = 8
    config.save_path = './vary_KC/files/' + str(i).zfill(2)

    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [50, 200, 800, 2500, 5000, 10000, 20000]
    return config, hp_ranges

def dup(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.N_ORN = 50
    config.N_PN = 50
    config.N_ORN_DUPLICATION = 10
    config.max_epoch = 5
    config.save_path = './duplication/files/' + str(i).zfill(2)
    hp_ranges = OrderedDict()
    hp_ranges['nothing'] = [0]
    return config, hp_ranges

def vary_kc_bias(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot'
    config.N_ORN = 50
    config.N_PN = 50
    config.N_ORN_DUPLICATION = 10
    config.max_epoch = 5
    config.save_path = './vary_KC_bias/files/' + str(i).zfill(2)
    hp_ranges = OrderedDict()
    hp_ranges['kc_bias'] = [-4,-3,-2,-1,0]
    return config, hp_ranges

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

def noise_pn_layer(i):
    config = configs.FullConfig()
    config.save_path = './why_pn_layer/no_noise_skip/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 10
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.max_epoch = 8
    config.train_pn2kc = False
    config.train_kc_bias = True
    config.sign_constraint_pn2kc = True
    config.sparse_pn2kc = True

    config.skip_orn2pn = False
    config.direct_glo = False
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    # hp_ranges['direct_glo'] = [True]
    hp_ranges['skip_orn2pn'] = [True]
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

def whatthef(i):
    config = configs.FullConfig()
    config.save_path = './random/files/' + str(i).zfill(2)
    config.data_dir = '../datasets/proto/nodup'
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.max_epoch = 15

    config.skip_orn2pn = True
    config.direct_glo = False

    config.kc_norm_post = None
    config.sign_constraint_pn2kc = True
    config.train_kc_bias = False
    config.kc_loss = True

    # config.initial_pn2kc = 0.5
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.mean_subtract_pn2kc = False
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    # hp_ranges['N_KC'] = [50, 100, 200, 400, 800, 1600, 2500, 5000, 10000, 20000]
    hp_ranges['initial_pn2kc'] = [.2]
    return config, hp_ranges

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    for i in range(0,100):
        tools.varying_config(whatthef, i)
