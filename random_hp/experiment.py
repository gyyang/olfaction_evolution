"""Randomly generate various models to train."""

import os
import argparse

import numpy as np

import configs
import tools
import train


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--device', help='CUDA device number', default=0, type=int)
parser.add_argument('-s0', '--seed0', help='Starting seed', default=0, type=int)
parser.add_argument('-s1', '--seed1', help='Ending seed', default=10, type=int)
args = parser.parse_args()


def train_random_model(i):
    config = configs.FullConfig()
    config.save_path = './files/random_hp_mlp/' + str(i).zfill(6)
    config.data_dir = './datasets/proto/_100_generalization_onehot_s0'
    config.max_epoch = 30
    config.ORN_NOISE_STD = 0.5
    config.N_ORN_DUPLICATION = 10
    config.model = 'normmlp'

    rng = np.random.RandomState(seed=i)
    neurons = np.exp(rng.uniform(np.log(10), np.log(5000), size=(10,))).astype(int)
    n_layer = rng.choice([1, 2, 3])
    config.NEURONS = [int(neurons[j]) for j in range(n_layer)]
    print(config.NEURONS)

    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    for i in range(args.seed0, args.seed1):
        print('[***] Hyper-parameter: %5d' % i)
        train_random_model(i)