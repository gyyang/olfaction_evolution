import os
from collections import defaultdict
import pickle
import json
import time

import tensorflow as tf

import task
from model import SingleLayerModel, FullModel
import tools


def make_input(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element


def train(config):
    tf.reset_default_graph()

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # Save config (first convert to dictionary)
    tools.save_config(config, save_path=config.save_path)

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    batch_size = config.batch_size
    n_batch = train_x.shape[0] // batch_size

    if config.model == 'full':
        CurrentModel = FullModel
    elif config.model == 'singlelayer':
        CurrentModel = SingleLayerModel
    else:
        raise ValueError('Unknown model type ' + str(config.model))

    # Build train model
    train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
    train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
    train_iter, next_element = make_input(train_x_ph, train_y_ph, batch_size)
    model = CurrentModel(next_element[0], next_element[1], config=config)

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    val_model = CurrentModel(val_x_ph, val_y_ph, config=config, is_training=False)

    # Make custom logger
    log = defaultdict(list)
    log_name = os.path.join(config.save_path, 'log.pkl')  # Consider json instead of pickle

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                    train_y_ph: train_y})

        loss = 0
        total_time, start_time = 0, time.time()
        for ep in range(config.max_epoch):
            # Validation
            val_loss, val_acc = sess.run([val_model.loss, val_model.acc],
                                         {val_x_ph: val_x, val_y_ph: val_y})
            print('[*] Epoch {:d}  train_loss={:0.2f}, val_loss={:0.2f}'.format(ep, loss, val_loss))
            print('Validation accuracy', val_acc[1])
            w_orn = sess.run(model.w_orn)
            glo_score, _ = tools.compute_glo_score(w_orn)
            print('Glo score ' + str(glo_score))

            if ep > 0:
                time_spent = time.time() - start_time
                total_time += time_spent
                print('Time taken {:0.1f}s'.format(total_time))
                print('Examples/second {:d}'.format(int(train_x.shape[0]/time_spent)))
            start_time = time.time()

            # Logging
            log['epoch'].append(ep)
            log['glo_score'].append(glo_score)
            log['train_loss'].append(loss)
            log['val_loss'].append(val_loss)
            log['val_acc'].append(val_acc[1])
            with open(log_name, 'wb') as f:
                pickle.dump(log, f, protocol=pickle.HIGHEST_PROTOCOL)

            # Train
            for b in range(n_batch):
                loss, _ = sess.run([model.loss, model.train_op])

        print('Training finished')
        model.save_pickle()
        model.save()


if __name__ == '__main__':
    experiment = 'robert'
    # experiment = 'peter'
    if experiment == 'peter':
        class modelConfig():
            dataset = 'repeat'
            model = 'singlelayer'
            N_ORN = 30
            lr = .001
            max_epoch = 100
            batch_size = 256
            save_path = './files/peter_tmp'

    elif experiment == 'robert':
        class modelConfig():
            dataset = 'proto'
            model = 'full'
            save_path = './files/peter_duplicate'
            N_ORN = task.PROTO_N_ORN * task.PROTO_N_ORN_PER_PN
            N_GLO = 50
            N_KC = 2500
            N_CLASS = task.PROTO_N_CLASS
            N_COMBINATORIAL_CLASS = task.N_COMBINATORIAL_CLASSES
            lr = .001
            max_epoch = 12
            batch_size = 256
            # Whether PN --> KC connections are sparse
            sparse_pn2kc = True
            # Whether PN --> KC connections are trainable
            train_pn2kc = False
            # Whether to have direct glomeruli-like connections
            direct_glo = False
            # Whether the coefficient of the direct glomeruli-like connection
            # motif is trainable
            train_direct_glo = True
            # Whether to tradeoff the direct and random connectivity
            tradeoff_direct_random = False
            # Whether to impose all cross area connections are positive
            sign_constraint = True
            # Whether to have layer-norm for KC
            kc_layernorm = True
            # dropout
            kc_dropout = True
            # label type can be either combinatorial, one_hot, sparse
            label_type = 'one_hot'
            data_dir = './datasets/proto/_threshold_onehot'
    else:
        raise NotImplementedError

    config = modelConfig()
    train(config)


