# if 'random_kc_claws' in experiments:
#     path = './files/random_kc_claws'
#     if TRAIN:
#         local_train(se.random_claw_configs(is_test), path)
#     if ANALYZE:
#         analysis_pn2kc_training.plot_distribution(path)
# analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
# sa.plot_weights(path, var_name='w_glo', sort_axis=-1, dir_ix=0)
# analysis_pn2kc_random.plot_distribution(path)
# analysis_pn2kc_random.claw_distribution(path, 'random')
# analysis_pn2kc_random.plot_cosine_similarity(path, 'preserve', log=False)
# analysis_pn2kc_random.pair_distribution(path, 'preserve')

if 'or2orn_primordial' in experiments:
    path = './files/or2orn_primordial'
    if TRAIN:
        local_train(experiments_receptor.primordial(is_test), path)
    if ANALYZE:
        sa.plot_weights(path, var_name = 'w_or', dir_ix=0, sort_axis=0)
        sa.plot_weights(path, var_name = 'w_or', dir_ix=1, sort_axis=0)
        sa.plot_weights(path, var_name = 'w_orn', sort_axis= 1, dir_ix=0)
        sa.plot_weights(path, var_name = 'w_orn', sort_axis= 1, dir_ix=1)
        sa.plot_weights(path, var_name = 'w_combined', dir_ix=0, sort_axis=0)
        sa.plot_weights(path, var_name = 'w_combined', dir_ix=1, sort_axis=0)

def primordial(argTest=False):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/primordial'
    config.max_epoch = 30

    config.receptor_layer = True
    config.or2orn_normalization = True
    config.pn_norm_pre = 'batch_norm'

    config.replicate_orn_with_tiling = True
    config.N_ORN_DUPLICATION = 10
    config.ORN_NOISE_STD = 0

    config.train_head1 = False
    config.train_head2 = True
    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['ORN_NOISE_STD'] = [0, .25]

    if argTest:
        config.max_epoch = testing_epochs
    return config, hp_ranges

def random_claw_configs(argTest=False):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.skip_orn2pn = True

    config.train_kc_bias=False
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.initial_pn2kc = .1

    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['dummy_var'] = [True]

    if argTest:
        config.max_epoch = testing_epochs
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
    hp_ranges['train_pn2kc'] = [True, True, False]
    hp_ranges['sparse_pn2kc'] = [False, False, True]
    hp_ranges['train_kc_bias'] = [False, False, True]
    hp_ranges['kc_loss'] = [False, True, False]
    hp_ranges['initial_pn2kc'] = [.1, .1, 0]

    if argTest:
        config.max_epoch = testing_epochs
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

def vary_pn2kc_initial_value_configs(argTest=False):
    '''
    Train from PN2KC with different initial connection values using constant initialization.
    Results:
        Claw count of ~7 should be independent of weight initialization

    '''
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 10

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.ORN_NOISE_STD = 0
    config.skip_orn2pn = True

    config.train_kc_bias=False
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.initial_pn2kc = .1
    config.save_every_epoch = True

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['initial_pn2kc'] = [.01, .02, .04, .08, .15, .3, .5, .7, 1, 1.5, 2]

    if argTest:
        config.max_epoch = testing_epochs
        hp_ranges['initial_pn2kc'] = [.05, .1, .25, .5, 1]

    return config, hp_ranges

if 'vary_pn2kc_initial_value' in experiments:
    path = './files/vary_pn2kc_initial_value'
    if TRAIN:
        local_train(experiments_controls_pn2kc.vary_pn2kc_initial_value_configs(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.plot_pn2kc_claw_stats(path, x_key='initial_pn2kc', dynamic_thres=True)
        analysis_pn2kc_training.plot_distribution(path)
        analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)
        sa.plot_results(path, x_key='initial_pn2kc', y_key='val_acc')

if 'vary_pn2kc_loss' in experiments:
    path = './files/vary_pn2kc_loss'
    if TRAIN:
        local_train(experiments_controls_pn2kc.vary_pn2kc_loss_configs(is_test), path)
    if ANALYZE:
        analysis_pn2kc_training.image_pn2kc_parameters(path)
        analysis_pn2kc_training.plot_distribution(path)
        sa.plot_results(path, x_key='kc_loss_beta', y_key='glo_score', loop_key='kc_loss_alpha')
        sa.plot_results(path, x_key='kc_loss_beta', y_key='val_acc', loop_key='kc_loss_alpha')

