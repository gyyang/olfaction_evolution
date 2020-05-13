from collections.__init__ import OrderedDict
import task
from configs import FullConfig, MetaConfig
from tools import vary_config
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

testing_epochs = 16


def standard(argTest=False):
    """Standard training setting"""
    config = FullConfig()
    config.max_epoch = 100

    config.pn_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.initial_pn2kc = 0.1
    config.save_every_epoch = False

    config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]
    if argTest:
        config.max_epoch = testing_epochs

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def receptor(argTest=False):
    """Standard training setting with full network including receptors."""
    config = FullConfig()
    config.max_epoch = 50

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.replicate_orn_with_tiling= True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0.2

    config.kc_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True

    config.data_dir = './datasets/proto/standard'
    config_ranges = OrderedDict()
    config_ranges['sign_constraint_orn2pn'] = [True]
    if argTest:
        config.max_epoch = 16

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def rnn(argTest=False):
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.model = 'rnn'

    config.NEURONS = 2500
    config.BATCH_NORM = False

    config.dropout = True
    config.dropout_rate = 0
    config.DIAGONAL = True

    config_ranges = OrderedDict()
    config_ranges['TIME_STEPS'] = [1, 2, 3]
    config_ranges['replicate_orn_with_tiling'] = [False, True, True]
    config_ranges['N_ORN_DUPLICATION'] = [1, 10, 10]
    if argTest:
        config.max_epoch = 16
    configs = vary_config(config, config_ranges, mode='sequential')
    return configs


def metalearn(argTest=False):
    config = MetaConfig()
    config.meta_lr = .001
    config.N_CLASS = 10 #10
    config.save_every_epoch = True
    config.meta_batch_size = 32 #32
    config.meta_num_samples_per_class = 8 #16
    config.meta_print_interval = 500

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.output_max_lr = 2.0 #2.0

    config.metatrain_iterations = 15000
    config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.initial_pn2kc = 0.05 #0.05

    config.data_dir = './datasets/proto/meta_dataset'

    config_ranges = OrderedDict()
    config_ranges['dummy'] = [True]

    if argTest:
        pass
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_pn(argTest=False):
    '''
    Vary number of PNs while fixing KCs to be 2500
    Results:
        GloScore should peak at PN=50, and then drop as PN > 50
        Accuracy should plateau at PN=50
        Results should be independent of noise
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    config.train_pn2kc = True
    config.sparse_pn2kc = False

    config_ranges = OrderedDict()
    config_ranges['N_PN'] = [10, 20, 30, 40, 50, 75, 100, 150, 200, 500, 1000]

    if argTest:
        config.max_epoch = testing_epochs

    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_kc(argTest=False):
    '''
    Vary number of KCs while also training ORN2PN.
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.pn_norm_pre = 'batch_norm'

    config.train_pn2kc = True
    config.sparse_pn2kc = False

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    config_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000, 20000]

    if argTest:
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_kc_activity_fixed(argTest):
    #TODO: use this one or the other one below
    '''

    :param argTest:
    :return:
    '''

    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    # config.train_pn2kc = True
    # config.sparse_pn2kc = False
    # config.initial_pn2kc = .1
    # config.extra_layer = True
    # config.extra_layer_neurons = 200

    config_ranges = OrderedDict()
    config_ranges['kc_dropout_rate'] = [0, .5]
    x = [100, 200, 500, 1000, 2000, 5000]
    config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]
    if argTest:
        config_ranges['kc_dropout_rate'] = [0, .5]
        x = [100, 200, 500, 1000]
        config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def vary_kc_activity_trainable(argTest):
    '''

    :param argTest:
    :return:
    '''
    config = FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.direct_glo = True
    config.pn_norm_pre = 'batch_norm'
    config.save_every_epoch = True

    config.train_pn2kc = True
    config.sparse_pn2kc = False
    # config.extra_layer = True
    # config.extra_layer_neurons = 200

    config_ranges = OrderedDict()
    config_ranges['kc_dropout_rate'] = [0, .5]
    x = [100, 200, 500, 1000, 2000, 5000]
    config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]
    if argTest:
        config_ranges['kc_dropout_rate'] = [.5]
        x = [100, 200, 500, 1000]
        config_ranges['data_dir'] = ['./datasets/proto/' + str(i) + '_100' for i in x]
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def pn_normalization(argTest):
    '''
    Assesses the effect of PN normalization on glo score and performance
    Results:

    :param argTest:
    :return:
    '''
    config = FullConfig()
    config.max_epoch = 15

    config.direct_glo = True
    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1

    # Ranges of hyperparameters to loop over
    config_ranges = OrderedDict()
    i = [0, .6]
    datasets = ['./datasets/proto/concentration_mask_row_' + str(s) for s in i]
    config_ranges['data_dir'] = ['./datasets/proto/standard'] + ['./datasets/proto/concentration'] + datasets
    config_ranges['pn_norm_pre'] = ['None','biology','fixed_activity']
    if argTest:
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def train_multihead(argTest=False):
    '''

    '''
    # config = configs.input_ProtoConfig()
    # config.label_type = 'multi_head_sparse'
    # task.save_proto(config, folder_name='multi_head')

    config = FullConfig()
    config.max_epoch = 30
    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.train_pn2kc = True
    config.sparse_pn2kc = False

    config.save_every_epoch = False
    # config.initial_pn2kc = .1
    # config.train_kc_bias = False
    # config.kc_loss = False

    config.data_dir = './datasets/proto/multi_head'

    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    config_ranges['dummy'] = [0, 1, 2]
    if argTest:
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs


def train_multihead_pruning(argTest=False):
    '''

    '''

    config = FullConfig()
    config.max_epoch = 30
    config.batch_size = 256
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.train_pn2kc = True
    config.sparse_pn2kc = False

    config.save_every_epoch = False
    config.initial_pn2kc = 10./config.N_PN
    config.kc_prune_threshold = 2./config.N_PN
    config.kc_prune_weak_weights = True

    config.data_dir = './datasets/proto/multi_head'

    config_ranges = OrderedDict()
    config_ranges['pn_norm_pre'] = [None, 'batch_norm']
    config_ranges['lr'] = [5e-3, 2e-3, 1e-3, 5*1e-4, 2*1e-4, 1e-4]
    config_ranges['dummy'] = [0, 1, 2]
    if argTest:
        config.max_epoch = testing_epochs
    configs = vary_config(config, config_ranges, mode='combinatorial')
    return configs