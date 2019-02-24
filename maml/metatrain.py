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

import configs
import tools
from dataset import DataGenerator
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
    batch_size = FLAGS.num_samples_per_class  # for classification should multiply by # classes
    data_generator = DataGenerator(
        batch_size=batch_size*2,
        meta_batch_size=FLAGS.meta_batch_size)

    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config

    model = MAML(config)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        tf.global_variables_initializer().run()
        tf.train.start_queue_runners()

        SUMMARY_INTERVAL = 100
        PRINT_INTERVAL = 1000

        print('Done initializing, starting training.')
        prelosses, postlosses = [], []

        for itr in range(FLAGS.metatrain_iterations):
            batch_x, batch_y = data_generator.generate()
            feed_dict = {model.input: batch_x, model.label: batch_y}

            input_tensors = [model.metatrain_op]

            if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
                input_tensors.extend([model.total_loss1, model.total_loss2])

            result = sess.run(input_tensors, feed_dict)

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
    train(config)

if __name__ == "__main__":
    main()
