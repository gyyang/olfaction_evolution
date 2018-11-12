"""A collection of experiments."""

import os
import task
import train
import configs
from collections import OrderedDict
import numpy as np

def vary_pn(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot_s0'
    config.N_ORN = 50
    config.N_ORN_DUPLICATION = 1
    config.max_epoch = 5
    config.save_path = './vary_PN/' + str(i).zfill(2)

    hp_ranges = OrderedDict()
    hp_ranges['N_PN_PER_ORN'] = list(range(10, 110, 10)) + [150, 200, 250]
    return config, hp_ranges

def vary_kc(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_0noise'
    config.N_ORN = 50
    config.N_ORN_DUPLICATION = 1
    config.N_PN_PER_ORN = 1
    config.max_epoch = 5
    config.save_path = './vary_KC/' + str(i).zfill(2)

    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [50, 100, 200, 400, 800, 1200, 2500, 5000, 10000, 20000]
    return config, hp_ranges



def varying_config(i):
    # Ranges of hyperparameters to loop over
    config, hp_ranges = vary_pn(i)

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
    for i in range(0,100):
        varying_config(i)
    # varying_config(8)