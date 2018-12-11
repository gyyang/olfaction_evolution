import os

import train
import configs
import task

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_config():
    config = configs.FullConfig()
    config.max_epoch = 30
    config.ORN_NOISE_STD = 0.1
    # simulate connecting directly to KC
    config.NEURONS = [2500]
    config.model = 'normmlp'
    config.data_dir = './datasets/proto/one_hidden_layer'
    config.save_path = './files/one_hidden_layer'
    return config


def train_one_layer():
    task_config = task.input_ProtoConfig()
    task.save_proto(config=task_config, seed=0, folder_name='one_hidden_layer')
    train.train(get_config())


if __name__ == '__main__':
    train_one_layer()
