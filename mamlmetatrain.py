"""
Run from the olfaction_evolution directory
python maml/metatrain.py
"""
import os
import time

import tensorflow as tf
from tensorflow.python.platform import flags

from train import make_input
import configs
import tools
from mamldataset import load_data
from mamlmodel import MAML
from mamldataset import DataGenerator
import numpy as np

# ## Training options
# config = flags.config
# flags.DEFINE_integer('metatrain_iterations', 100000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
# flags.DEFINE_integer('meta_batch_size', 16, 'number of tasks sampled per meta-update')
# flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
# flags.DEFINE_integer('num_samples_per_class', 8, 'number of examples used for inner gradient update (K for K-shot learning).')
# flags.DEFINE_float('update_lr', .3, 'step size alpha for inner gradient update.') # 0.1 for omniglot
# flags.DEFINE_integer('num_updates', 1, 'number of inner gradient updates during training.')
# 
# ## Model options
# flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
# flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

LOAD_DATA = False
def print_results(res):
    # TODO: need cleaning
    n_steps = len(res[1])
    print('Pre-update train loss >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(res[0], res[3][0],res[3][1]))
    print('Post-update train loss >> {:0.4f}.  acc >> {:0.2f}, {:0.2f} '.format(res[2], res[5][0],res[5][1]))
    print('Post-update val loss step 1 >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(res[1][0], res[4][0][0],res[4][0][1]))
    print('Post-update val loss step {:d} >> {:0.4f}.  acc >> {:0.2f}, {:0.2f}'.format(n_steps, res[1][-1],res[4][-1][0],res[4][-1][1]))

def train_from_path(path):
    """Train from a path with a config file in it."""
    config = tools.load_config(path)
    train(config)

def train(config):
    tf.reset_default_graph()

    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    tools.save_config(config, save_path=config.save_path)

    PRINT_INTERVAL = config.meta_print_interval
    if LOAD_DATA:
        train_x, train_y = load_data('./datasets/proto/meta_proto')
        # Build train model
        train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
        train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
        train_iter, next_element = make_input(
            train_x_ph, train_y_ph, config.meta_batch_size)
        model = MAML(next_element[0], next_element[1], config)
    else:
        num_samples_per_class = config.meta_num_samples_per_class
        dim_output = config.N_CLASS  # TODO: this doesn't have to be
        num_class = config.meta_labels_per_class * dim_output
        data_generator = DataGenerator(
            dataset= config.data_dir,
            batch_size=num_samples_per_class * num_class * 2,
            meta_batch_size=config.meta_batch_size,
            num_samples_per_class=num_samples_per_class,
            num_class=num_class,
            dim_output=dim_output,
        )
        train_x_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                                 data_generator.batch_size,
                                                 config.N_ORN))

        if config.label_type == 'one_hot':
            train_y_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                                     data_generator.batch_size,
                                                     dim_output))
        elif config.label_type == 'multi_head_one_hot':
            train_y_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                                     data_generator.batch_size,
                                                     dim_output + config.n_class_valence))
        else:
            raise ValueError('label type {} is not recognized'.format(config.label_type))
        model = MAML(train_x_ph, train_y_ph, config)

    model.summ_op = tf.summary.merge_all()

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True

    # n_save_every = 10
    # ind_orn = []
    # for i in range(50):
    #     ind_orn += list(range(i, 500, 50))
    # ind_orn = np.array(ind_orn)
    # weight_layer1 = []
    # weight_layer2 = []

    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if LOAD_DATA:
            sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                        train_y_ph: train_y})

        train_writer = tf.summary.FileWriter(config.save_path, sess.graph)
        print('Done initializing, starting training.')

        total_time, start_time = 0, time.time()
        for itr in range(config.metatrain_iterations):
            if model.metatrain_op_lr is not None:
                input_tensors = [model.metatrain_op, model.metatrain_op_lr]
                ix = 2
            else:
                input_tensors = [model.metatrain_op]
                ix = 1

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


            # if itr % n_save_every == 0:
            #     w_orn, w_glo = sess.run([model.model.w_orn, model.model.w_glo])
            #     weight_layer1.append(w_orn[ind_orn])
            #     weight_layer2.append(w_glo[:,:30])

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
                lr = sess.run(model.update_lr)
                print('Learned update_lr is : ' +
                      np.array2string(lr, precision=2, separator=',', suppress_small=True))
                print_results(res[ix:])
                model.save_pickle(itr)
                model.save()
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

        # np.save(os.path.join(config.save_path, 'w_layer1'), weight_layer1)
        # np.save(os.path.join(config.save_path, 'w_layer2'), weight_layer2)
        model.save_pickle()
        model.save()

def main():
    import shutil
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    config = configs.MetaConfig()
    config.metatrain_iterations = 30000
    config.meta_lr = .001
    config.N_PN = 50
    config.N_CLASS = 5
    config.meta_labels_per_class = 1
    config.meta_batch_size = 16
    config.meta_num_samples_per_class = 8
    config.meta_print_interval = 500

    config.replicate_orn_with_tiling = False
    config.N_ORN_DUPLICATION = 1
    config.train_orn2pn = False
    config.direct_glo = True
    # config.pn_norm_pre = 'batch_norm'
    config.kc_norm_pre = 'batch_norm'

    config.train_kc_bias = True
    config.sparse_pn2kc = False
    config.train_pn2kc = True
    config.save_path = './files/metalearn/0'

    # config.data_dir = './datasets/proto/multi_head'
    # config.label_type = 'multi_head_one_hot'
    # config.data_dir = './datasets/proto/test'
    config.data_dir = './datasets/proto/test_norn_200'
    config.label_type = 'one_hot'

    try:
        shutil.rmtree(config.save_path)
    except FileNotFoundError:
        pass
    train(config)

if __name__ == "__main__":
    main()
