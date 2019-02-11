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