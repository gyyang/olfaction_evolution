"""A collection of experiments."""

import os
import task
import train
import configs

def varying_config(i):
    # Ranges of hyperparameters to loop over
    noise = [0, .25, .5]

    if i >= len(noise):
        return

    config = configs.FullConfig()
    config.save_path = './files_' + str(i).zfill(2)
    config.N_ORN_PER_PN = 10
    config.ORN_NOISE_STD = noise[i]

    data_path = task.save_proto(config)
    config.data_dir = data_path
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(1,100):
        varying_config(i)