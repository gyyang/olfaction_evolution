if 'kcrole' in experiments:
    # Compare with or without KC layer
    path = './files/kcrole'
    if TRAIN:
        train(standard.experiment_controls.train_kcrole(is_test), path, sequential=True)
    if ANALYZE:
        # evaluatewithnoise.evaluate_kcrole(path, 'weight_perturb')
        evaluatewithnoise.plot_kcrole(path, 'weight_perturb')

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