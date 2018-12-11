import os

import task
import train
import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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