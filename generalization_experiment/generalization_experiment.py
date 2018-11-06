"""A collection of experiments."""

import os
from collections import OrderedDict

import numpy as np

import task
import train


def varying_config(i):
    class modelConfig():
        dataset = 'proto'
        model = 'full'
        save_path = None
        N_ORN = task.PROTO_N_ORN
        N_GLO = 50
        N_KC = 2500
        N_CLASS = task.PROTO_N_CLASS
        N_COMBINATORIAL_CLASS = task.N_COMBINATORIAL_CLASSES
        lr = .001
        max_epoch = 10
        batch_size = 256
        # Whether PN --> KC connections are sparse
        sparse_pn2kc = True
        # Whether PN --> KC connections are trainable
        train_pn2kc = False
        # Whether to have direct glomeruli-like connections
        direct_glo = False
        # Whether the coefficient of the direct glomeruli-like connection
        # motif is trainable
        train_direct_glo = True
        # Whether to tradeoff the direct and random connectivity
        tradeoff_direct_random = False
        # Whether to impose all cross area connections are positive
        sign_constraint = True
        # dropout
        kc_dropout = True
        # label type can be either combinatorial, one_hot, sparse
        label_type = 'one_hot'
        data_dir = './datasets/proto/_threshold_one-hot'
        generalization_percent = 100


    config = modelConfig()
    config.save_path = './files/generalization_experiment/' + str(i).zfill(2)

    # Ranges of hyperparameters to loop over

    generalization_interval = 10
    generalization_percent = generalization_interval * i
    config.generalization_percent = generalization_percent
    if generalization_percent > 100:
        return
    task.save_proto_hard(argThreshold=1, argCombinatorial=0, percent_generalization=generalization_percent)
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(100):
        varying_config(i)