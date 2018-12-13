import os
from collections import OrderedDict

import tools
import task
import train
import configs

def vary_train_pn2kc_initial_value_configs(argTest=False):
    '''
    Train (with loss) from PN2KC while skipping ORN2PN with different initial connection values
    using uniform initialization.
    Results:
        Claw count of 6-7 should be independent of weight initialization

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['initial_pn2kc'] = [.01, .02, .04, .08, .15, .3, .5, 1]

    if argTest:
        config.max_epoch = 10
        hp_ranges['initial_pn2kc'] = [.01, .04, .3, 1]
    return config, hp_ranges


def vary_kc_loss_alpha_configs(argTest=False):
    '''
    Train (with loss) from PN2KC while skipping ORN2PN with different KC loss strengths.
    Results:
        Claw count of 6-7 should be relatively independent of KC loss strength

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.initial_pn2kc = 0

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_loss_alpha'] = [.1, .3, 1, 3, 10, 30, 100]
    hp_ranges['kc_loss_beta'] = [.1, .3, 1, 3, 10, 30, 100]
    #exclude based on accuracy, based on nKC with no connections

    if argTest:
        config.max_epoch = 5
        hp_ranges['kc_loss_alpha'] = [.1, 1, 10, 100]
        hp_ranges['kc_loss_beta'] = [.1, 1, 10, 100]
    return config, hp_ranges