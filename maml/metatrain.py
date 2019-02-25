"""
python main.py --metatrain_iterations=70000 --norm=None --num_samples_per_class=10
"""
import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

# rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(rootpath)  # TODO: This is hacky, should be fixed

from train import make_input
import configs
import tools
from dataset import load_data
from maml import MAML

FLAGS = flags.FLAGS

## Training options
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('num_samples_per_class', 5, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 1e-3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')


def train(config):
    tf.reset_default_graph()

    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config

    train_x, train_y = load_data(None, './datasets/proto/meta_proto')

    # Build train model
    train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
    train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
    train_iter, next_element = make_input(
        train_x_ph, train_y_ph, FLAGS.meta_batch_size)

    model = MAML(next_element[0], next_element[1], config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                    train_y_ph: train_y})

        SUMMARY_INTERVAL = 100
        PRINT_INTERVAL = 1000

        print('Done initializing, starting training.')
        prelosses, postlosses = [], []

        for itr in range(FLAGS.metatrain_iterations):
            input_tensors = [model.metatrain_op]

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.total_loss1, model.total_loss2])

            result = sess.run(input_tensors)

            if itr % SUMMARY_INTERVAL == 0:
                prelosses.append(result[-2])
                postlosses.append(result[-1])

            if (itr != 0) and itr % PRINT_INTERVAL == 0:
                print_str = 'Iteration ' + str(itr)
                print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(
                    np.mean(postlosses))
                print(print_str)
                prelosses, postlosses = [], []

def main():
    config = configs.FullConfig()
    config.N_KC = 2500
    train(config)

if __name__ == "__main__":
    main()
