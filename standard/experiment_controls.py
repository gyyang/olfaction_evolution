from collections import OrderedDict

import numpy as np

from configs import FullConfig
from tools import vary_config


testing_epochs = 12


def control_nonnegative():
    """Standard training setting"""
    config = FullConfig()
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'
    config.data_dir = './datasets/proto/standard'

    config_ranges = OrderedDict()
    config_ranges['sign_constraint_orn2pn'] = [True, False]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_orn2pn():
    '''
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.pn_norm_pre = 'batch_norm'

    config.train_pn2kc = False

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30]
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['ORN_NOISE_STD']= [0, .1, .2]
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_pn2kc_backup():
    '''
    This is the setup Peter last used
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = [0.05, 0.1, 0.2, 0.5]
    config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_pn2kc():
    '''
    New setup Robert using for torch models
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200

    config.N_ORN_DUPLICATION = 1
    config.direct_glo = True  # skip_orn2pn has same effect
    config.pn_norm_pre = 'batch_norm'

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10./config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    config_ranges['lr'] = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['train_kc_bias'] = [False, True]
    config_ranges['initial_pn2kc'] = np.array([2., 5., 10., 20.])/config.N_PN
    # config_ranges['apl'] = [False, True]

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_pn2kc_inhibition():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200

    config.N_ORN_DUPLICATION = 1
    # config.direct_glo = True
    config.skip_orn2pn = True
    config.pn_norm_pre = 'batch_norm'

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    # config.kc_prune_weak_weights = True
    config.initial_pn2kc = 5./config.N_PN
    config.kc_prune_threshold = 1./config.N_PN

    config.kc_recinh = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_prune_weak_weights'] = [True, False]
    config_ranges['kc_recinh_coeff'] = list(np.arange(0, 1.01, 0.2))
    config_ranges['kc_recinh_step'] = list(range(1, 10, 2))
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_prune_boolean(n_pn=50):
    """Standard training setting"""
    config = FullConfig()
    config.max_epoch = 200

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.pn_norm_pre = 'batch_norm'
    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./n_pn

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    # Heuristics
    if n_pn > 100:
        config.lr = 1e-3
    else:
        config.lr = 2e-3

    config_ranges = OrderedDict()
    config_ranges['kc_prune_weak_weights'] = [False, True]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_pn2kc_prune_hyper(n_pn=50):
    """Standard training setting"""
    # New setup Robert using for torch models
    config = FullConfig()
    config.max_epoch = 200

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.pn_norm_pre = 'batch_norm'

    config.kc_prune_weak_weights = True
    config.kc_prune_threshold = 1./n_pn

    # New settings
    config.batch_size = 8192  # Much bigger batch size
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'  # Prevent degeneration
    config.lr = 2e-3

    config_ranges = OrderedDict()
    config_ranges['lr'] = [5e-2, 2e-2, 1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4]
    config_ranges['N_KC'] = [2500, 5000, 10000]
    config_ranges['kc_prune_threshold'] = np.array([0.5, 1., 2.])/n_pn
    config_ranges['initial_pn2kc'] = np.array([2.5, 5, 10., 20.])/n_pn
    configs = vary_config(config, config_ranges, mode='control')
    return configs


def control_vary_kc():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [50, 100, 200, 400, 1000, 2500, 5000, 10000, 20000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_vary_pn():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    config_ranges = OrderedDict()
    config_ranges['N_PN'] = [20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]
    config_ranges['kc_dropout_rate'] = [0, 0.25, 0.5]
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs

#TODO
def controls_receptor():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100

    config.train_pn2kc = False

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.pn_norm_pre = 'batch_norm'
    config.ORN_NOISE_STD = .4

    config.replicate_orn_with_tiling = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30,100]
    config_ranges['or2orn_normalization'] = [False, True]
    config_ranges['pn_norm_pre'] = ['None', 'batch_norm']

    configs = vary_config(config, config_ranges, mode='control')
    return configs


def vary_init_sparse():
    """Vary if initialization is dense or sparse"""
    config = FullConfig()
    config.max_epoch = 100
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    config.pn_norm_pre = 'batch_norm'

    config.data_dir = './datasets/proto/standard'
    config.save_every_epoch = True
    config_ranges = OrderedDict()
    config_ranges['initializer_pn2kc'] = ['constant', 'single_strong']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_apl():
    """Vary APL."""
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True

    config.save_every_epoch = True

    config_ranges = OrderedDict()
    config_ranges['apl'] = [False, True]
    config_ranges['kc_norm_pre'] = [None, 'batch_norm']

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def pn_normalization_direct():
    '''
    Assesses the effect of PN normalization on glo score and performance
    '''
    config = FullConfig()
    config.skip_orn2pn = True
    config.train_pn2kc = False
    config.max_epoch = 30
    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['data_dir'] = ['./datasets/proto/concentration_mask',
                             # './datasets/proto/standard',
                             ]
    config_ranges['pn_norm_post'] = ['custom', 'None']

    # TODO: hyperparameter search
    # try:
    #     rmax = tools.load_pickle(path, 'model/layer1/r_max:0')
    #     print('rmax: {}'.format(rmax))
    #     rho = tools.load_pickle(path, 'model/layer1/rho:0')
    #     print('rho: {}'.format(rho))
    #     m = tools.load_pickle(path, 'model/layer1/m:0')
    #     print('m: {}'.format(m))
    # except:
    #     pass
    #
    # try:
    #     gamma = tools.load_pickle(path, 'model/layer1/LayerNorm/gamma:0')
    #     print('gamma params: {}'.format(gamma))
    # except:
    #     pass
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def train_claw_configs():
    '''
    Train (with or without loss) or fix connections from PN2KC while skipping ORN2PN
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['train_pn2kc'] = [True, False]
    config_ranges['sparse_pn2kc'] = [False, True]
    config_ranges['train_kc_bias'] = [False, True]

    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_claw_configs():
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.train_pn2kc = False

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm' #not necessary, but for standardization

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['kc_inputs'] = list(range(1,15, 2)) + list(range(15,30, 3)) + \
                             list(range(30, 50, 5))
    config_ranges['ORN_NOISE_STD'] = [0, 0.25, 0.5]

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def control_n_or_per_orn():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200
    config.pn_norm_pre = 'batch_norm'

    config.batch_size = 8192
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.lr = 2 * 1e-3
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'

    config_ranges = OrderedDict()
    config_ranges['n_or_per_orn'] = list(range(1, 10))
    config_ranges['data_dir'] = ['./datasets/proto/n_or_per_orn'+str(n)
                             for n in range(1, 10)]
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def vary_orn_corr():
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 200
    config.pn_norm_pre = 'batch_norm'

    config.batch_size = 8192
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.lr = 2 * 1e-3
    config.initial_pn2kc = 10. / config.N_PN
    config.initializer_pn2kc = 'uniform'

    orn_corrs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    datasets = ['./datasets/proto/orn_corr_{:0.2f}'.format(c) for c in orn_corrs]

    config_ranges = OrderedDict()
    config_ranges['orn_corr'] = orn_corrs
    config_ranges['data_dir'] = datasets
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs
