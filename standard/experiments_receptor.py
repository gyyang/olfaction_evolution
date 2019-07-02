from collections import OrderedDict

import task
import configs


testing_epochs = 8

def basic(argTest=False):
    #TODO: figure out a way to get rid of normalization. batch_norm fails in the condition of 0 noise
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.save_every_epoch = True
    config.pn_norm_pre = 'batch_norm'

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['ORN_NOISE_STD'] = [0, .25]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_normalization(argTest=False):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.receptor_layer = True

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.NOISE_MODEL = 'additive'
    config.ORN_NOISE_STD = 0

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['or2orn_normalization'] = [False, True]
    hp_ranges['orn2pn_normalization'] = [False, True]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def vary_receptor_duplication(argTest=False):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.orn2pn_normalization = True
    # config.pn_norm_pre = 'batch_norm'

    config.replicate_orn_with_tiling = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30, 100, 300]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def vary_hyper_parameters(argTest=False):
    #TODO: vary hyper-paramaeter values for Wilson/Olsen normalization
    #TODO: check that the hyper-parameter values are similar to biology

    pass
