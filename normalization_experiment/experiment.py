"""A collection of experiments."""

import os
from collections import OrderedDict

import numpy as np
import configs
import train



def varying_config(i):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/_50_generalization_onehot'
    config.save_path = './files/' + str(i)
    config.max_epoch = 10

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['pn_norm_pre_nonlinearity'] = ['batch_norm', 'subtract_norm', None]

    # Unravel the input index
    keys = hp_ranges.keys()
    dims = [len(hp_ranges[k]) for k in keys]
    n_max = np.prod(dims)

    if i >= n_max:
        # do nothing
        return

    indices = np.unravel_index(i % n_max, dims=dims)

    # Set up new hyperparameter
    for key, index in zip(keys, indices):
        setattr(config, key, hp_ranges[key][index])
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # for i in range(1,100):
    #     varying_config(i)
    varying_config(2)