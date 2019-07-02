import tools
import train
import os
import mamlmetatrain

def basic_train(experiment, save_path):
    config = experiment()
    config.save_path = save_path
    train.train(config)


def local_train(experiment, save_path, train_arg = None):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))


def local_sequential_train(experiment, save_path, train_arg = None):
    """Train all models locally."""
    for i in range(0, 1000):
        config = tools.varying_config_sequential(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))

def local_control_train(experiment, save_path, train_arg = None):
    '''
    Train each hyper-parameter separately
    '''
    for i in range(0, 1000):
        config = tools.varying_config_control(experiment, i)
        if config:
            print('[***] Hyper-parameter: %2d' % i)
            config.save_path = os.path.join(save_path, str(i).zfill(6))

            if train_arg == None:
                train.train(config)
            elif train_arg == 'metalearn':
                mamlmetatrain.train(config)
            else:
                raise ValueError('training function is not recognized by keyword {}'.format(train_arg))
