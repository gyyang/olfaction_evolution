import tools
import train
import os


def basic_train(experiment, save_path):
    config = experiment()
    config.save_path = save_path
    train.train(config)


def local_train(experiment, save_path):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)


def local_sequential_train(experiment, save_path):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))
            train.train(config)