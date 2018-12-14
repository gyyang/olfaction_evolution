"""A collection of experiments."""

import os
import task
import configs
from collections import OrderedDict
import numpy as np
import tools
import train

import standard.experiment as se
import standard.experiment_controls_pn2kc as experiments_controls_pn2kc
from standard.hyper_parameter_train import local_train, local_sequential_train
import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import standard.analysis_pn2kc_random as analysis_pn2kc_random


def temp():
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 8
    config.replicate_orn_with_tiling = False
    config.skip_orn2pn = True
    config.train_pn2kc = True
    config.sparse_pn2kc = False
    config.train_kc_bias = False
    config.kc_loss = True
    config.initial_pn2kc = 0.1
    config.save_every_epoch = True
    config.kc_loss_beta = 100
    config.kc_loss_alpha = 100

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['dummy'] = [True]
    return config, hp_ranges

path = './test/temp'
local_train(temp(), path)
analysis_pn2kc_training.plot_distribution(path)
analysis_pn2kc_training.plot_sparsity(path)