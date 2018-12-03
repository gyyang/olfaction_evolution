"""A collection of experiments."""

import os
import task
import train
import configs
from collections import OrderedDict
import numpy as np

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
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.N_ORN = 50
    config.N_PN = 50
    config.N_ORN_DUPLICATION = 10
    config.max_epoch = 5
    config.save_path = './vary_KC_bias/files/' + str(i).zfill(2)
    hp_ranges = OrderedDict()
    hp_ranges['kc_bias'] = [-4,-3,-2,-1,0]
    return config, hp_ranges

def vary_sparse_kc(i):
    config = configs.FullConfig()
    config.save_path = './sparse/dup_trainable/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 10
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.max_epoch = 10
    config.kc_norm_post = None
    # config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.sign_constraint_pn2kc = True
    config.skip_orn2pn = False
    config.direct_glo = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['sparse_pn2kc'] = [True,False]
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
    config.save_path = './no_pn_layer/vary_kc_claws_only_bias_untrainable/files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 1
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0
    config.data_dir = '../datasets/proto/_100_generalization_onehot'
    config.max_epoch = 8
    config.kc_norm_post = None
    config.train_pn2kc = False
    config.train_kc_bias = True
    config.sign_constraint_pn2kc = True
    config.skip_orn2pn = True
    config.direct_glo = False
    config.sparse_pn2kc = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    # hp_ranges['sparse_pn2kc'] = [True, False]
    hp_ranges['kc_inputs'] = [1, 8, 15, 50, 200]
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

# def noise_claws(i):
#     config = configs.FullConfig()
#     config.save_path = './why_pn_layer/trainable_claws_noise/files/' + str(i).zfill(2)
#     config.N_ORN_DUPLICATION = 10
#     config.N_ORN = 50
#     config.ORN_NOISE_STD = 0
#     config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
#     config.max_epoch = 8
#     config.train_pn2kc = True
#     config.train_kc_bias = True
#     config.sign_constraint_pn2kc = True
#     config.skip_orn2pn = True
#     config.sparse_pn2kc = False
#
#     # Ranges of hyperparameters to loop over
#     hp_ranges = OrderedDict()
#     hp_ranges['direct_glo'] = [True, False]
#     return config, hp_ranges


def varying_config(experiment, i):
    # Ranges of hyperparameters to loop over
    config, hp_ranges = experiment(i)

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)
    indices = np.unravel_index(i % n_max, dims=dims)

    if i >= n_max:
        return

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(0,50):
        print('[***] Hyper-parameter: %2d' % i)
        varying_config(noise_pn_layer, i)
