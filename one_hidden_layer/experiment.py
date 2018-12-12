import os

import train
import configs
import task

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def get_config():
    config = configs.FullConfig()
    config.max_epoch = 30
    config.ORN_NOISE_STD = 0.0
    config.N_ORN_DUPLICATION = 1
    config.N_PN = 500 # mammal has around 1000 at least
    config.N_KC = 10000
    # simulate connecting directly to KC
    #config.NEURONS = [10000]
    #config.model = 'normmlp'
    config.data_dir = './datasets/proto/one_hidden_layer'
    config.save_path = './files/one_hidden_layer'
    return config


def train_one_layer():
    network_config = get_config()
    task_config = task.input_ProtoConfig()
    task_config.N_ORN = network_config.N_PN * network_config.N_ORN_DUPLICATION
    task.save_proto(config=task_config, seed=0, folder_name='one_hidden_layer')
    train.train(network_config)


if __name__ == '__main__':
    train_one_layer()
