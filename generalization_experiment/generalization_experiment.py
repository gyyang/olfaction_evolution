"""A collection of experiments."""

import os
import task
import train
import model


def varying_config(i):
    # Ranges of hyperparameters to loop over
    generalization_interval = 10
    generalization_percent = generalization_interval * i
    if generalization_percent > 100:
        return

    config = model.FullConfig()
    config.save_path = './files_' + str(i).zfill(2)
    config.percent_generalization = generalization_percent
    data_path = task.save_proto(config)
    config.data_dir = data_path
    train.train(config)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for i in range(7,11,3):
        varying_config(i)