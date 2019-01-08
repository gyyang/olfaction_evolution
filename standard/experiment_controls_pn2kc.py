import os
from collections import OrderedDict

import tools
import task
import train
import configs

testing_epochs = 8

def vary_pn2kc_initial_value_configs(argTest=False):
    '''
    Train (with loss) from PN2KC while skipping ORN2PN with different initial connection values
    using uniform initialization.
    Results:
        Claw count of 6-7 should be independent of weight initialization

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 10
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['initial_pn2kc'] = [.01, .02, .04, .08, .15, .3, .5, .7, 1, 1.5, 2]
    hp_ranges['kc_norm_pre'] = [None]
    # hp_ranges['initializer_pn2kc'] = ['normal','constant']

    if argTest:
        config.max_epoch = 30
        hp_ranges['initial_pn2kc'] = [1]

    return config, hp_ranges

def vary_pn2kc_noise_configs(argTest=False):
    '''
    Train (with loss) from PN2KC with perfect ORN2PN connectivity, while varying levels of noise + dup
    Results:
        Claw count of 6-7 should be independent of noise and number of ORN duplicates

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 10
    config.replicate_orn_with_tiling = True
    config.direct_glo = True
    config.skip_orn2pn = False

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['ORN_NOISE_STD'] = [0, .2, .4, .6, .8, 1]

    if argTest:
        config.max_epoch = 8
        hp_ranges['ORN_NOISE_STD'] = [0, .5, 1]
        hp_ranges['N_ORN_DUPLICATION'] = [1, 10, 100]
    return config, hp_ranges

def vary_pn2kc_loss_configs(argTest=False):
    '''
    Train (with loss) from PN2KC while skipping ORN2PN with different KC loss strengths.
    Results:
        Claw count of 6-7 should be relatively independent of KC loss strength

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 10
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.initial_pn2kc = 0.1

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_loss_alpha'] = [.1, .3, 1, 3, 10, 30, 100]
    hp_ranges['kc_loss_beta'] = [.1, .3, 1, 3, 10, 30, 100]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_loss_alpha'] = [.1, .3, 1, 3, 10, 30, 100]
        hp_ranges['kc_loss_beta'] = [.1, .3, 1, 3, 10, 30, 100]
    return config, hp_ranges

def vary_kc_dropout_configs(argTest):
    '''
    Vary KC dropout. Observe the requirement for nKC as a function of dropout
    Results:

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.replicate_orn_with_tiling = True
    config.ORN_NOISE_STD = 0
    config.skip_orn2pn = False

    config.kc_loss = False
    config.kc_loss_alpha = 1
    config.kc_loss_beta = 10

    config.train_pn2kc = False
    config.sparse_pn2kc = True
    config.train_kc_bias = True
    config.initial_pn2kc = 0
    config.save_every_epoch = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_dropout'] = [0, .1, .3, .5, .7, .9]
    hp_ranges['N_KC'] = [50, 100, 200, 400, 800, 1600, 2500, 5000, 10000]

    if argTest:
        config.max_epoch=testing_epochs
        hp_ranges['kc_dropout'] = [0, .25, .5, .75]

    return config, hp_ranges

def vary_initialization_method_configs(argTest):
    '''
    Vary initialization scheme to be constant or normal as a function of KC claw count
    Results:
        Accuracy should depend on the initialization condition for fixed weights
        Accuracy should be low for claws > 40 for uniform initialization, as negative bias will not cancel correlations
        Accuracy should be high for claws > 50 for normal initialization, as negative bias will cancel out correlations
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = list(range(1,15, 2)) + list(range(15,30, 3)) + \
                             list(range(30, 50, 5))
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [3, 5, 7, 9, 11, 15, 20, 30, 40, 50]
        hp_ranges['initializer_pn2kc'] = ['constant','normal']
    return config, hp_ranges
    pass