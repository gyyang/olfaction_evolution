"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

def local_train(experiment, save_path):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)

def temp(_):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/standard'
    config.replicate_orn_with_tiling = True
    config.ORN_NOISE_STD = 0.5
    config.max_epoch = 5

    config.skip_orn2pn = False
    config.direct_glo = False
    config.initializer_orn2pn = 'normal'

    config.train_kc_bias = True
    config.kc_loss = False

    config.initial_pn2kc = 0
    config.initializer_pn2kc = 'uniform'
    config.train_pn2kc = False

    config.sparse_pn2kc = True
    config.kc_inputs = 7

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [3]
    return config, hp_ranges

if __name__ == '__main__':
    local_train(temp, './test/files')
    tools.load_all_results(',/')
