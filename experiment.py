"""A collection of experiments."""

import task
import train


experiment = 'robert'
class modelConfig():
    dataset = 'proto'
    model = 'full'
    save_path = './files/robert_dev'
    N_ORN = task.PROTO_N_ORN
    N_GLO = 50
    N_KC = 2500
    N_CLASS = task.PROTO_N_CLASS
    lr = .001
    max_epoch = 5
    batch_size = 256
    # Whether PN --> KC connections are sparse
    sparse_pn2kc = True
    # Whether PN --> KC connections are trainable
    train_pn2kc = False
    # Whether to have direct glomeruli-like connections
    direct_glo = True
    # Whether the coefficient of the direct glomeruli-like connection
    # motif is trainable
    train_direct_glo = True
    # Whether to tradeoff the direct and random connectivity
    tradeoff_direct_random = False
    # Whether to impose all cross area connections are positive
    sign_constraint = True
    # dropout
    kc_dropout = True


config = modelConfig()
train.train(config)