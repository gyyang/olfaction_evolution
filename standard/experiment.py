import os
from collections import OrderedDict

import tools
import task
import train
import configs

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train_orn2pn(save_path):
    task_config = task.input_ProtoConfig()
    task.save_proto(config=task_config, seed=0, folder_name='standard')

    config = configs.FullConfig()
    config.max_epoch = 30
    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.data_dir = './datasets/proto/standard'
    config.save_path = save_path
    train.train(config)


def train_orn2pn2kc():
    # TODO: add check if dataset already exists
    # TODO(pw): to be finished
    config = configs.FullConfig()
    config.max_epoch = 30
    config.sparse_pn2kc = True
    config.train_pn2kc = False
    config.data_dir = './datasets/proto/standard'
    config.save_path = './files/standard/orn2pn'
    train.train(config)


def local_train(experiment, save_path):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)


def vary_kc_configs(_):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [50, 100, 200, 300, 400, 500, 1000, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    return config, hp_ranges


def vary_n_orn_duplication(_):
    config = configs.FullConfig()
    config.data_dir = './datasets/proto/standard'
    config.max_epoch = 30
    config.ORN_NOISE_STD = 0.5

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [100, 2500]
    hp_ranges['N_ORN_DUPLICATION'] = [1, 3, 5, 7, 10]
    return config, hp_ranges

def temp(_):
    config = configs.FullConfig()
    config.data_dir = '../datasets/proto/standard'
    config.max_epoch = 8

    # Ranges of hyperparameters to loop over
    hp_ranges = OrderedDict()
    hp_ranges['N_KC'] = [300, 2500, 10000]
    hp_ranges['ORN_NOISE_STD'] = [0, 0.5, 1.0]
    return config, hp_ranges

if __name__ == '__main__':
    local_train(temp, '../files/test')