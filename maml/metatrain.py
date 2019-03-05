"""
python main.py --metatrain_iterations=70000 --norm=None --num_samples_per_class=10
"""
import os
import sys
import time
print(sys.path)
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

from train import make_input
import configs
import tools
from maml.dataset import load_data
from maml.maml import MAML

FLAGS = flags.FLAGS

## Training options
flags.DEFINE_integer('metatrain_iterations', 100000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 64, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('num_samples_per_class', 20, 'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_float('update_lr', 0.1, 'step size alpha for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')

## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

LOAD_DATA = False

def print_results(res):
    print('Pre-update train loss {:0.4f}  acc {:0.2f}'.format(res[0], res[3]))
    print('Post-update train loss {:0.4f}  acc {:0.2f}'.format(res[2], res[5]))
    print('Post-update val loss step 1 {:0.4f}  acc {:0.2f}'.format(res[1][0],
                                                                    res[4][0]))
    print('Post-update val loss step 5 {:0.4f}  acc {:0.2f}'.format(res[1][-1],
                                                                    res[4][
                                                                        -1]))

def train(config):
    tf.reset_default_graph()

    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config

    if LOAD_DATA:
        train_x, train_y = load_data(None, './datasets/proto/meta_proto')
        # Build train model
        train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
        train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
        train_iter, next_element = make_input(
            train_x_ph, train_y_ph, FLAGS.meta_batch_size)

        model = MAML(next_element[0], next_element[1], config)
    else:
        from maml.dataset import DataGenerator
        num_samples_per_class = FLAGS.num_samples_per_class
        num_class = config.n_class_valence * 2  # TODO: this doesn't have to be
        dim_output = config.n_class_valence
        data_generator = DataGenerator(
            batch_size=num_samples_per_class * num_class * 2,
            meta_batch_size=FLAGS.meta_batch_size,
            num_samples_per_class=num_samples_per_class,
            num_class=num_class,
            dim_output=dim_output,
        )
        train_x_ph = tf.placeholder(tf.float32, (FLAGS.meta_batch_size,
                                                 data_generator.batch_size,
                                                 config.N_PN))
        train_y_ph = tf.placeholder(tf.float32, (FLAGS.meta_batch_size,
                                                 data_generator.batch_size,
                                                 dim_output))
        model = MAML(train_x_ph, train_y_ph, config)

    model.summ_op = tf.summary.merge_all()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if LOAD_DATA:
            sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                        train_y_ph: train_y})

        PRINT_INTERVAL = 1000
        train_writer = tf.summary.FileWriter(config.save_path, sess.graph)
        print('Done initializing, starting training.')

        total_time, start_time = 0, time.time()
        for itr in range(FLAGS.metatrain_iterations):
            input_tensors = [model.metatrain_op]

            if itr % PRINT_INTERVAL == 0:
                input_tensors.extend(
                    [model.total_loss1, model.total_loss2, model.total_loss3,
                     model.total_acc1, model.total_acc2, model.total_acc3,
                     model.summ_op])

            try:
                if LOAD_DATA:
                    res = sess.run(input_tensors)
                else:
                    train_x, train_y = data_generator.generate('train')
                    res = sess.run(input_tensors,
                                   {train_x_ph: train_x, train_y_ph: train_y})
            except KeyboardInterrupt:
                print('Training interrupted by users')
                break

            # if itr % SUMMARY_INTERVAL == 0:
            #     prelosses.append(result[-2])
            #     postlosses.append(result[-1])

            if itr % PRINT_INTERVAL == 0:
                print('Iteration ' + str(itr))
                if itr > 0:
                    time_spent = time.time() - start_time
                    total_time += time_spent
                    print('Time taken {:0.1f}s'.format(total_time))
                    # TODO: this calculation is wrong
                    print('Examples/second {:d}'.format(int(train_x.shape[0]/time_spent)))
                start_time = time.time()

                print('Meta-train')
                print_results(res[1:])
                model.save_pickle(itr)
                train_writer.add_summary(res[-1], itr)

                if not LOAD_DATA:
                    val_x, val_y = data_generator.generate('val')
                    input_tensors = [
                        model.total_loss1, model.total_loss2, model.total_loss3,
                        model.total_acc1, model.total_acc2, model.total_acc3]
                    res = sess.run(input_tensors,
                                   {train_x_ph: val_x, train_y_ph: val_y})

                    print('Meta-val')
                    print_results(res)

        model.save_pickle()

def main():
    import shutil

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = configs.FullConfig()
    config.N_KC = 2500
    config.n_class_valence = 2
    config.kc_dropout = True
    config.sign_constraint_pn2kc = True
    config.sparse_pn2kc = False
    config.save_path = './files/metatrain/valence2/0'
    config.data_dir = './datasets/proto/standard'
    try:
        shutil.rmtree(config.save_path)
    except FileNotFoundError:
        pass
    train(config)

if __name__ == "__main__":
    main()
