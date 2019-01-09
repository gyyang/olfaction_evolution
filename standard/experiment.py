import os
from collections import OrderedDict

import task
import configs

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

testing_epochs = 6
def train_orn2pn(argTest=False):
    '''
    Most basic experiment. Train ORN2PN.
    Result:
        Show that GloScore increases as a function of training
    '''
    task_config = task.input_ProtoConfig()
    task.save_proto(config=task_config, seed=0, folder_name='standard')

    config = configs.FullConfig()
    config.max_epoch = 30
    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.data_dir = './datasets/proto/standard'
    hp_ranges = OrderedDict()
    hp_ranges['dummy_var'] = [True]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def vary_orn_duplication_configs(argTest=False):
    '''
    Vary the number of ORN duplicates
    Results:
        GloScore should be robust to duplicates
        Accuracy should increase when there are more copies of ORNs to deal with noise
        Accuracy should be higher for more KCs
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.ORN_NOISE_STD = 0.5

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [100, 2500]
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 5, 7, 10]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['N_KC'] = [2500]
        hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10]

    return config, hp_ranges

def vary_pn_configs(argTest=False):
    '''
    Vary number of PNs while fixing KCs to be 2500
    Results:
        GloScore should peak at PN=50, and then drop as PN > 50
        Accuracy should plateau at PN=50
        Results should be independent of noise
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    hp_ranges = OrderedDict()
    hp_ranges['N_PN'] = [10, 20, 30, 40, 50, 75, 100, 150, 200, 500]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['N_PN'] = [20, 50, 200]
        hp_ranges['ORN_NOISE_STD'] = [0, .5]

    return config, hp_ranges

def vary_kc_configs(argTest=False):
    '''
    Vary number of KCs while also training ORN2PN.
    Results:
        GloScore and Accuracy peaks at >2500 KCs for all noise values
        GloScore does not depend on noise. Should be lower for higher noise values
        GloScore depends on nKC. Should be lower for lower nKC
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['N_KC'] = [200, 2500, 10000]
        hp_ranges['ORN_NOISE_STD'] = [0, 0.5]
    return config, hp_ranges

def vary_claw_configs(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.skip_orn2pn = False
    config.direct_glo = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = list(range(1,15, 2)) + list(range(15,30, 3)) + \
                             list(range(30, 50, 5))
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [3, 7, 11, 15, 20, 35, 50]
        hp_ranges['ORN_NOISE_STD'] = [0, 0.25]
    return config, hp_ranges

def train_claw_configs(argTest=False):
    '''
    NOTE: this should be trained with varying_config_sequential

    Train (with or without loss) or fix connections from PN2KC while skipping ORN2PN
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['train_pn2kc'] = [True, True, False]
    hp_ranges['sparse_pn2kc'] = [False, False, True]
    hp_ranges['train_kc_bias'] = [False, False, True]
    hp_ranges['kc_loss'] = [False, True, False]
    hp_ranges['initial_pn2kc'] = [.1, .1, 0]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def random_claw_configs(argTest=False):
    '''
    NOTE: this should be trained with varying_config_sequential

    Train (with or without loss) or fix connections from PN2KC while skipping ORN2PN
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.save_every_epoch = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.initial_pn2kc = .1

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['dummy_var'] = [True]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def train_orn2pn2kc_configs(argTest):
    '''
    Allow both ORN2PN and PN2KC connections to be trained simultaneously
    Results:
        Glo score increases
        Claw count converges to ~7
        Results should be independent of noise
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.replicate_orn_with_tiling = True
    config.skip_orn2pn = False

    config.kc_loss = False
    config.kc_loss_alpha = 1
    config.kc_loss_beta = 10

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.initial_pn2kc = 0
    config.save_every_epoch = True
    config.initializer_pn2kc = 'normal'

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['ORN_NOISE_STD'] = [0, .5, 1]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['ORN_NOISE_STD'] = [0]
    return config, hp_ranges

def vary_norm(argTest):
    '''
    Vary normalization methods
    Results:

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.train_pn2kc = False
    config.sparse_pn2kc = True
    config.train_kc_bias = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['pn_norm_pre'] = [None, 'batch_norm', 'layer_norm']
    hp_ranges['kc_norm_pre'] = [None, 'batch_norm', 'layer_norm']

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def train_multihead(argTest=False):
    '''

    '''
    config = configs.input_ProtoConfig()
    config.label_type = 'multi_head_sparse'
    task.save_proto(config, folder_name='multi_head')

    config = configs.FullConfig()
    config.max_epoch = 30
    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    # config.initial_pn2kc = .1
    # config.train_kc_bias = False
    # config.kc_loss = False

    config.pn_norm_pre = 'batch_norm'
    config.data_dir = './datasets/proto/multi_head'
    config.save_every_epoch = True

    hp_ranges = OrderedDict()
    hp_ranges['dummy_var'] = [True]
    if argTest:
        config.max_epoch = testing_epochs

    return config, hp_ranges


def train_multihead_sequential():
    config = configs.input_ProtoConfig()
    config.label_type = 'multi_head_sparse'
    task.save_proto(config, folder_name='multi_head')

    import train
    config = configs.FullConfig()

    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.train_pn2kc = True
    config.sparse_pn2kc = False

    config.pn_norm_pre = 'batch_norm'
    config.data_dir = './datasets/proto/multi_head'
    config.save_path = './files/multihead_sequential/0'
    config.save_every_epoch = True

    config.max_epoch = 10
    config.train_head1 = False
    train.train(config)

    config.max_epoch = 30
    config.train_head1 = True
    train.train(config, reload=True)


def temp(argTest):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/standard'
    config.max_epoch = 8

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [300, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    return config, hp_ranges