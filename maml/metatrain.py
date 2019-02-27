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

        PRINT_INTERVAL = 100

        print('Done initializing, starting training.')
        prelosses, postlosses = [], []

        for itr in range(FLAGS.metatrain_iterations):
            input_tensors = [model.metatrain_op]

            if itr % PRINT_INTERVAL == 0:
                input_tensors.extend(
                    [model.total_loss1, model.total_loss2, model.total_loss3,
                     model.total_acc1, model.total_acc2, model.total_acc3])

            try:
                res = sess.run(input_tensors)
            except KeyboardInterrupt:
                print('Training interrupted by users')
                break

            # if itr % SUMMARY_INTERVAL == 0:
            #     prelosses.append(result[-2])
            #     postlosses.append(result[-1])

            if itr % PRINT_INTERVAL == 0:
                print('Iteration ' + str(itr))
                print('Pre loss {:0.4f}  acc {:0.2f}'.format(res[1], res[4]))
                print('Post train loss {:0.4f}  acc {:0.2f}'.format(res[3], res[6]))
                print('First Post val loss {:0.4f}  acc {:0.2f}'.format(res[2][0], res[5][0]))
                print('Last Post val loss {:0.4f}  acc {:0.2f}'.format(res[2][-1], res[5][-1]))
                prelosses, postlosses = [], []
                model.save_pickle(itr)

        model.save_pickle()

def main():
    import shutil
    try:
        shutil.rmtree('./files/tmp_metatrain/')
    except FileNotFoundError:
        pass
    config = configs.FullConfig()
    config.N_KC = 50
    config.n_class_valence = 2
    config.sign_constraint_pn2kc = False
    config.save_path = './files/tmp_metatrain/0'
    train(config)

if __name__ == "__main__":
    main()
