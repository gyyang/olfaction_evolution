"""A collection of experiments."""

import os
import task
import train
import configs
from collections import OrderedDict
import numpy as np

def varying_config(i):
    config = configs.FullConfig()
    config.save_path = './files/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 10
    config.N_PN_PER_ORN

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['ORN_NOISE_STD'] = [0, .25, .5]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)

    if i >= n_max:
        return

    indices = np.unravel_index(i % n_max, dims=dims)
    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    train.train(config)

    data_path = task.save_proto(config)
    config.data_dir = data_path
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(0,100):
        varying_config(i)