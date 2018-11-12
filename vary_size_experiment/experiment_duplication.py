"""A collection of experiments."""

import os
import task
import train
import configs
from collections import OrderedDict
import numpy as np

def varying_config(i):
    config = configs.FullConfig()
    config.save_path = './vary_PN_dupORN10x_.25Noise/' + str(i).zfill(2)
    config.N_ORN_DUPLICATION = 10
    config.N_ORN = 50
    config.ORN_NOISE_STD = 0.25
    config.data_dir = '../datasets/proto/_100_generalization_onehot_dup_.25noise'
    config.max_epoch = 10
    config.kc_norm_post = None
    config.N_PN_PER_ORN = 1
    config.train_pn2kc = False

    train.train(config)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    varying_config(0)

