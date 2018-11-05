"""A collection of experiments."""

import os
from collections import OrderedDict

import numpy as np

import task
import train


def varying_config(i):
    class modelConfig():
        dataset = 'proto'
        model = 'full'
        save_path = None
        N_ORN = task.PROTO_N_ORN
        N_GLO = 50
        N_KC = 2500
        N_CLASS = task.PROTO_N_CLASS
        lr = .001
        max_epoch = 10
        batch_size = 256
        # Whether PN --> KC connections are sparse
        sparse_pn2kc = True
        # Whether PN --> KC connections are trainable
        train_pn2kc = False
        # Whether to have direct glomeruli-like connections
        direct_glo = True
        # Whether the coefficient of the direct glomeruli-like connection
        # motif is trainable
        train_direct_glo = True
        # Whether to tradeoff the direct and random connectivity
        tradeoff_direct_random = False
        # Whether to impose all cross area connections are positive
        sign_constraint = True
        # dropout
        kc_dropout = True

    config = modelConfig()
    config.save_path = './files/vary_config/' + str(i)

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['sparse_pn2kc'] = [True, False]
    hp_ranges['train_pn2kc'] = [True, False]
    hp_ranges['direct_glo'] = [True, False]
    hp_ranges['sign_constraint'] = [True, False]


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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    for i in range(100):
        varying_config(i)