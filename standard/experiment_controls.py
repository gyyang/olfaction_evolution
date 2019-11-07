from collections import OrderedDict
from collections.__init__ import OrderedDict

import numpy as np

import configs
from standard.experiment import testing_epochs

testing_epochs = 12

def controls_glomeruli(argTest=False):
    '''
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 60
    config.pn_norm_pre = 'batch_norm'
    config.ORN_NOISE_STD = 0


    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 10, 30]
    hp_ranges['pn_norm_pre'] = ['None', 'batch_norm']
    hp_ranges['ORN_NOISE_STD']= [0, .1, .2]
    hp_ranges['kc_dropout_rate'] = [0, .25, .5, .75]
    hp_ranges['lr'] = [3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


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


def vary_lr_standard(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    # config.initial_pn2kc = 10./n_pn

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_lr_n_kc(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_lr_n_kc_batchnorm(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.pn_norm_pre = 'batch_norm'

    config.save_every_epoch = False

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_new_lr_n_kc(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.initial_pn2kc = 10./n_pn # heuristic

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_pn2kc_init(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 500

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.N_KC = 5000

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [1e-3, 5*1e-4, 2*1e-4, 1e-4, 5*1e-5]
    hp_ranges['initial_pn2kc'] = np.array([2., 4., 10.])/n_pn
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_prune_pn2kc_init(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 200

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    # config.N_KC = 5000

    config.initial_pn2kc = 10./n_pn

    config.kc_prune_weak_weights = True
    # config.kc_prune_threshold = 0.02

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    # hp_ranges['N_KC'] = [2500, 5000, 10000, 20000]
    hp_ranges['N_KC'] = [10000, 20000]
    hp_ranges['kc_prune_threshold'] = np.array([1., 2., 5.])/n_pn
    # hp_ranges['initial_pn2kc'] = np.array([2., 4., 10.])/n_pn
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_prune_lr(argTest=False, n_pn=50):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 30

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn'+str(n_pn)

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.N_KC = min(n_pn**2, 40000)

    config.initial_pn2kc = 10./n_pn

    config.kc_prune_weak_weights = True

    config.save_every_epoch = False
    config.save_log_only = True

    hp_ranges = OrderedDict()
    lr_range_dict = {
            25: (0.005, 0.01),
            50: (0.002, 0.005),
            75: (0.002, 0.005),
            100: (0.001, 0.002),
            150: (0.0005, 0.001),
            200: (0.0002, 0.0005),
            400: (0.0002, 0.0005)
            }

    try:
        lr_range = lr_range_dict[n_pn]
    except KeyError:
        lr_range = (1e-4, 5e-3)

    hp_ranges['lr'] = np.logspace(np.log10(lr_range[0]), np.log10(lr_range[1]), 10)
    hp_ranges['kc_prune_threshold'] = np.array([0.5, 1., 2., 5.])/n_pn
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_n_orn_longtrain(argTest=False):
    """Standard training setting"""
    config = configs.FullConfig()
    config.max_epoch = 200

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.save_every_epoch = True

    hp_ranges = OrderedDict()
    n_pns = [50, 200]
    hp_ranges['N_PN'] = n_pns
    hp_ranges['data_dir'] = ['./datasets/proto/orn'+str(n) for n in n_pns]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def temp(argTest):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/standard'
    config.max_epoch = 8

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [300, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    return config, hp_ranges


def vary_init_sparse(argTest=False):
    """Vary if initialization is dense or sparse"""
    config = configs.FullConfig()
    config.max_epoch = 100
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    config.pn_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.data_dir = './datasets/proto/standard'
    config.save_every_epoch = True
    hp_ranges = OrderedDict()
    hp_ranges['initializer_pn2kc'] = ['constant', 'single_strong']
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_init_sparse_lr(argTest=False, n_pn=50):
    """Vary if initialization is dense or sparse"""
    config = configs.FullConfig()
    config.max_epoch = 100

    config.N_PN = n_pn
    config.data_dir = './datasets/proto/orn' + str(n_pn)
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0

    config.pn_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.initial_pn2kc = 10./config.N_PN

    config.save_every_epoch = False
    config.save_log_only = False
    hp_ranges = OrderedDict()
    hp_ranges['lr'] = [1e-3, 5 * 1e-4, 2 * 1e-4, 1e-4]
    hp_ranges['initializer_pn2kc'] = ['constant', 'single_strong']
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_apl(argTest=False):
    """Vary APL."""
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.save_every_epoch = True

    hp_ranges = OrderedDict()
    hp_ranges['apl'] = [False, True]
    hp_ranges['kc_norm_pre'] = [None, 'batch_norm']
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def vary_w_glo_meansub_coeff(argTest=False):
    """Vary APL."""
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 5

    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.

    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.skip_orn2pn = True
    config.w_glo_meansub = True
    config.kc_bias = 0.5

    config.save_every_epoch = True

    hp_ranges = OrderedDict()
    cs = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    hp_ranges['w_glo_meansub_coeff'] = cs
    hp_ranges['kc_bias'] = [-1 + 2*c for c in cs]
    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def kc_generalization(argTest=False):
    config = configs.FullConfig()
    config.batch_size = 4
    config.max_epoch = 500
    config.save_epoch_interval = 25
    config.save_every_epoch = True
    config.skip_orn2pn = True

    hp_ranges = OrderedDict()
    x = [100]
    hp_ranges['data_dir'] = ['./datasets/proto/small_training_set_' + str(i) for i in x] * 2
    hp_ranges['skip_pn2kc'] = [True, False]
    hp_ranges['replicate_orn_with_tiling'] = [True, False]
    hp_ranges['N_ORN_DUPLICATION'] = [50, 1]

    if argTest:
        pass
    return config, hp_ranges


def train_kcrole(argTest=False):
    '''
    NOTE: this should be trained with varying_config_sequential

    Compare networks with or without KC layer
    Results:
        Accuracy from training PN2KC weights = fixed PN2KC weights
        Accuracy from Training PN2KC weights with KC loss = without KC loss
        Training PN2KC weights with loss should result in KC claw count of 6-7
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.N_ORN_DUPLICATION = 1
    config.kc_dropout = False
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['train_pn2kc'] = [True, False, True]
    hp_ranges['sparse_pn2kc'] = [False, True, False]
    hp_ranges['skip_pn2kc'] = [False, False, True]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges


def pn_normalization_direct(argTest):
    '''
    Assesses the effect of PN normalization on glo score and performance
    Results:

    :param argTest:
    :return:
    '''
    config = configs.FullConfig()
    config.skip_orn2pn = True
    config.max_epoch = 30
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['data_dir'] = ['./datasets/proto/concentration_mask',
                             # './datasets/proto/standard',
                             ]
    hp_ranges['pn_norm_post'] = ['custom', 'None']
    if argTest:
        config.max_epoch = testing_epochs
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
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['train_pn2kc'] = [True, False]
    hp_ranges['sparse_pn2kc'] = [False, True]
    hp_ranges['train_kc_bias'] = [False, True]

    if argTest:
        config.max_epoch = testing_epochs
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

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm' #not necessary, but for standardization

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = list(range(1,15, 2)) + list(range(15,30, 3)) + \
                             list(range(30, 50, 5))
    hp_ranges['ORN_NOISE_STD'] = [0, 0.25, 0.5]
    if argTest:
        config.max_epoch = 70
        hp_ranges['kc_inputs'] = [1, 3, 5, 7, 10, 15, 20, 30]
        hp_ranges['ORN_NOISE_STD'] = [0]
    return config, hp_ranges


def vary_claw_configs_new(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    # TODO: Need to merge this with vary_claw_configs
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.kc_dropout = False
    config.save_every_epoch = True
    # config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'  # not necessary, but for standardization

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 25, 30]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.1]
    if argTest:
        config.max_epoch = 30
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
        hp_ranges['ORN_NOISE_STD'] = [0]
    return config, hp_ranges


def vary_claw_configs_long(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    # TODO: Need to merge this with vary_claw_configs
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 100
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.kc_dropout = False
    config.save_every_epoch = False
    # config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'  # not necessary, but for standardization

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [7, 15, 30]
    if argTest:
        config.max_epoch = 100
    return config, hp_ranges


def vary_claw_configs_dev(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    # TODO: Need to merge this with vary_claw_configs
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/normalized'
    config.max_epoch = 2

    config.lr = 0.001
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.train_kc_bias = False
    config.kc_dropout = False
    # config.output_bias = False
    config.batch_size = 256
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 30]
    # hp_ranges['kc_inputs'] = [3, 7, 30]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
    return config, hp_ranges


def vary_claw_configs_fixedacc(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    # TODO: Need to merge this with vary_claw_configs
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/tmp'
    config.max_epoch = 10

    config.lr = 0.001
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    # config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'

    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.train_kc_bias = False
    config.kc_dropout = False
    config.output_bias = False
    config.batch_size = 256
    config.save_every_epoch = False

    config.target_acc = 0.6

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [3, 5, 7, 9, 12, 15, 20, 30]
    # hp_ranges['kc_inputs'] = [3, 7, 30]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
    return config, hp_ranges


def vary_claw_configs_frequentevaluation(argTest=False):
    '''
    Vary number of inputs to KCs while skipping ORN2PN layer
    Results:
        Accuracy should be high at around claw values of 7-15
        # Noise dependence
    '''
    # TODO: Need to merge this with vary_claw_configs
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 50

    config.lr = 0.01
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0.
    config.skip_orn2pn = True
    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.kc_dropout = False
    config.batch_size = 100
    config.n_batch = 1
    config.save_every_epoch = False

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 30]*5
    # hp_ranges['kc_inputs'] = [3, 7, 30]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
    return config, hp_ranges


def vary_claw_configs_orn200(argTest=False):
    '''
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/orn200'
    config.max_epoch = 3
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 25, 30]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
    return config, hp_ranges


def vary_claw_configs_orn500(argTest=False):
    '''
    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/orn500'
    config.max_epoch = 2
    config.N_ORN_DUPLICATION = 1
    config.skip_orn2pn = True
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['kc_inputs'] = [1, 3, 5, 7, 9, 12, 15, 20, 25, 30]
    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['kc_inputs'] = [1, 3, 7, 10, 15, 30]
    return config, hp_ranges