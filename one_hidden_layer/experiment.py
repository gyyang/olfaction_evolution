import os

import train
import configs

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = configs.FullConfig()
config.max_epoch = 30
config.ORN_NOISE_STD = 0.0
# simulate connecting directly to KC
config.NEURONS = [2500]
config.model = 'normmlp'
config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
config.save_path = './files/one_hidden_layer'

if __name__ == '__main__':
    train.train(config)