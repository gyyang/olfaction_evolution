from collections.__init__ import OrderedDict

import configs
from standard.experiment import testing_epochs

testing_epochs = 12

def controls_kc_claw(argTest):
    '''
    Show that claw count of 7 is independent of dropout
    Results:

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.direct_glo = True

    config.pn_norm_pre = 'batch_norm'
    config.train_kc_bias=False
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.save_every_epoch = True
    config.kc_dropout = True
    config.kc_dropout_rate = .5

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_dropout_rate'] = [0, .1, .2, .3, .4, .5, .6]
    hp_ranges['ORN_NOISE_STD'] = [0, .05, .1, .15, .2]
    hp_ranges['pn_norm_pre'] = ['None', 'batch_norm']

    if argTest:
        # hp_ranges['kc_dropout_rate'] = [0, .2, .4, .6]
        config.max_epoch=testing_epochs
    return config, hp_ranges


def controls_glomeruli(argTest=False):
    '''
    Vary the number of ORN duplicates
    Results:
        GloScore should be robust to duplicates
        Accuracy should increase when there are more copies of ORNs to deal with noise
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'
    config.ORN_NOISE_STD = 0


    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30, 100]
    hp_ranges['pn_norm_pre'] = ['None', 'batch_norm']
    hp_ranges['ORN_NOISE_STD']= [0, .05, .1, .15, .2]
    hp_ranges['kc_dropout_rate'] = [0, .1, .2, .3, .4, .5, .6]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30, 100]
        hp_ranges['ORN_NOISE_STD'] = [0, .1, .2]
        hp_ranges['kc_dropout_rate'] = [0, .3, .6]

    return config, hp_ranges

def controls_receptor(argTest=False):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.pn_norm_pre = 'batch_norm'
    config.ORN_NOISE_STD = .4

    config.replicate_orn_with_tiling = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30,100]
    hp_ranges['or2orn_normalization'] = [False, True]
    hp_ranges['pn_norm_pre'] = ['None', 'batch_norm']

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges
