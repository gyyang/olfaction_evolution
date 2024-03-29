import os
import sys
from collections import defaultdict
import pickle
import json
import time

import numpy as np
import tensorflow as tf

import task
from model import SingleLayerModel, FullModel, NormalizedMLP, AutoEncoder, AutoEncoderSimple, RNN, ParameterizeK
from configs import FullConfig, SingleLayerConfig
import tools
from standard.analysis_pn2kc_training import _compute_sparsity


def make_input(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6))
    # Making sure the shape is fully defined
    try:
        data = data.batch(tf.cast(batch_size, tf.int64), drop_remainder=True)
    except TypeError:
        data = data.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    # data = data.batch(tf.cast(batch_size, tf.int64))
    data = data.repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element


def train_from_path(path):
    """Train from a path with a config file in it."""
    config = tools.load_config(path)
    train(config, reload=True)


def train(config, reload=False, save_everytrainloss=False):
    tf.reset_default_graph()

    # Merge model config with config from dataset
    dataset_config = tools.load_config(config.data_dir)
    dataset_config.update(config)
    config = dataset_config
    for item in config.__dict__.items():
        print(item)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)
    # Save config
    tools.save_config(config, save_path=config.save_path)

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

    batch_size = config.batch_size
    if 'n_batch' in dir(config):
        n_batch = config.n_batch
    else:
        n_batch = train_x.shape[0] // batch_size

    if config.model == 'full':
        CurrentModel = FullModel
    elif config.model == 'singlelayer':
        CurrentModel = SingleLayerModel
    elif config.model == 'normmlp':
        CurrentModel = NormalizedMLP
    elif config.model == 'oracle':
        CurrentModel = OracleNet
    elif config.model == 'autoencode':
        CurrentModel = AutoEncoder
        # CurrentModel = AutoEncoderSimple
    elif config.model == 'rnn':
        CurrentModel = RNN
    elif config.model == 'K':
        CurrentModel = ParameterizeK
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
    val_model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)

    if 'set_oracle' in dir(config) and config.set_oracle:
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype, [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = CurrentModel(oracle_x_ph, oracle_y_ph, config=config, training=False)

    # Make custom logger
    log = defaultdict(list)

    glo_score_mode = 'tile' if config.replicate_orn_with_tiling else 'repeat'

    # validation fetches
    val_fetch_names = ['loss', 'acc']
    try:
        _ = val_model.acc2
        val_fetch_names.append('acc2')
    except AttributeError:
        pass

    if config.model == 'full' and config.train_pn2kc:
        val_fetch_names.append('kc')

    val_fetches = [getattr(val_model, f) for f in val_fetch_names]

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.log_device_placement = False

    finish_training = False
    with tf.Session(config=tf_config) as sess:
        # sess.run(tf.global_variables_initializer())
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
        sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                    train_y_ph: train_y})


        start_epoch = 0
        if reload:
            try:
                model.load()
                log = tools.load_log(config.save_path)
                start_epoch = log['epoch'][-1] + 1
            except:
                print('No model file to be reloaded, starting anew')

        if 'set_oracle' in dir(config) and config.set_oracle:
            oracle.set_oracle_weights()

        if config.model == 'rnn' and config.DIAGONAL:
            model.set_weights()

        loss = 0
        acc = 0
        lr = config.lr
        acc_smooth = 0
        total_time, start_time = 0, time.time()
        w_bins = np.linspace(0, 1, 201)
        w_bins_log = np.linspace(-20, 5, 201)
        log['w_bins'] = w_bins
        log['w_bins_log'] = w_bins_log
        lin_bins = np.linspace(0, 1, 1001)
        log['lin_bins'] = lin_bins
        activity_bins = np.linspace(0, 1, 201)

        for ep in range(start_epoch, config.max_epoch):
            # Validation
            tmp = sess.run(val_fetches, {val_x_ph: val_x, val_y_ph: val_y})
            res = {name:r for name, r in zip(val_fetch_names, tmp)}

            if ep % config.save_epoch_interval == 0:
                print('[*' + '*'*50 + '*]')
                print('Epoch {:d}'.format(ep))
                print('Train/Validation loss {:0.2f}/{:0.2f}'.format(loss, res['loss']))
                print('Train/Validation accuracy {:0.2f}/{:0.2f}'.format(acc, res['acc']))
                print('Learning rate {:0.3E}'.format(lr))
                log['epoch'].append(ep)
                log['train_loss'].append(loss)
                log['train_acc'].append(acc)
                log['lr_now'].append(lr)
                for key, value in res.items():
                    if key in ['kc']:
                        continue
                    log['val_' + key].append(value)

                try:
                    print('Validation accuracy head 2 {:0.2f}'.format(res['acc2']))
                except KeyError:
                    pass

                if config.model == 'full':
                    if config.train_pn2kc:
                        w_glo = sess.run(model.w_glo)
                        w_glo[w_glo<1e-9] = 1e-9 #finite range for log
                        kcs = res['kc']

                        coding_level = (kcs > 0).mean()
                        coding_level_per_kc = kcs.mean(axis=0)
                        coding_level_per_odor = kcs.mean(axis=1)
                        log['coding_level'].append(coding_level)
                        hist, _ = np.histogram(coding_level_per_kc, bins=activity_bins)
                        log['coding_level_per_kc'].append(hist)
                        hist, _ = np.histogram(coding_level_per_odor, bins=activity_bins)
                        log['coding_level_per_odor'].append(hist)

                        # Store distribution of flattened weigths
                        log_hist, _ = np.histogram(np.log(w_glo.flatten()), bins=w_bins_log)
                        hist, _ = np.histogram(w_glo.flatten(), bins=w_bins)
                        log['log_hist'].append(log_hist)
                        log['hist'].append(hist)
                        log['kc_w_sum'].append(w_glo.sum(axis=0))
                        hist, _ = np.histogram(w_glo.flatten(), bins=lin_bins)
                        log['lin_hist'].append(hist)
                        # Store sparsity computed with threshold

                        sparsity_inferred, thres_inferred = _compute_sparsity(w_glo, dynamic_thres=True, thres=.1)
                        K_inferred = sparsity_inferred[sparsity_inferred > 0 ].mean()
                        bad_KC_inferred = np.sum(sparsity_inferred == 0) / sparsity_inferred.size
                        log['sparsity_inferred'].append(sparsity_inferred)
                        log['thres_inferred'].append(thres_inferred)
                        log['K_inferred'].append(K_inferred)
                        log['bad_KC_inferred'].append(bad_KC_inferred)

                        sparsity, thres = _compute_sparsity(w_glo, dynamic_thres=False, thres= config.kc_prune_threshold)
                        K = sparsity[sparsity > 0].mean()
                        bad_KC = np.sum(sparsity == 0)/sparsity.size
                        log['sparsity'].append(sparsity)
                        log['thres'].append(thres)
                        log['K'].append(K)
                        log['bad_KC'].append(bad_KC)

                        print('KC coding level={}'.format(np.round(coding_level,2)))
                        print('Bad KCs (fixed, inferred) ={}, {}'.format(bad_KC, bad_KC_inferred))
                        print('K (fixed, inferred) ={}, {}'.format(K, K_inferred))

                    if config.receptor_layer:
                        w_or = sess.run(model.w_or)
                        or_glo_score, _ = tools.compute_glo_score(
                            w_or, config.N_ORN, glo_score_mode)
                        print('OR receptor glo score ' + str(or_glo_score))
                        log['or_glo_score'].append(or_glo_score)

                        w_orn = sess.run(model.w_orn)
                        glo_score, _ = tools.compute_glo_score(
                            w_orn, config.N_ORN, 'matrix', w_or)
                        print('Glo score ' + str(glo_score))
                        log['glo_score'].append(glo_score)

                        w_combined = np.matmul(w_or, w_orn)
                        combined_glo_score, _ = tools.compute_glo_score(
                            w_combined, config.N_ORN, glo_score_mode)
                        print('Combined glo score ' + str(combined_glo_score))
                        log['combined_glo_score'].append(combined_glo_score)

                    else:
                        if config.train_orn2pn and not config.skip_orn2pn:
                            w_orn = sess.run(model.w_orn)
                            glo_score, _ = tools.compute_glo_score(
                                w_orn, config.N_ORN, glo_score_mode)
                            log['glo_score'].append(glo_score)
                            print('Glo score ' + str(glo_score))

                            sim_score, _ = tools.compute_sim_score(
                                w_orn, config.N_ORN, glo_score_mode)
                            log['sim_score'].append(sim_score)
                            print('Sim score ' + str(sim_score))
                        else:
                            w_orn = sess.run(model.w_orn)
                            glo_score, _ = tools.compute_glo_score(
                                w_orn, config.N_ORN, glo_score_mode)
                            log['glo_score'].append(glo_score)
                            print('Glo score ' + str(glo_score))

                elif config.model == 'K':
                    K = sess.run(model.K)
                    bias = sess.run(model.b_glo)
                    kcs = sess.run(model.kc)

                    log['K'].append(K)
                    print('K ={}'.format(np.round(K,2)))
                    print('Bias (mean)={}'.format(np.round(bias.mean(),2)))
                    print('kc activity (mean) = {}'.format(kcs.mean()))
                    print('kc sparseness (mean) = {}'.format(np.mean(kcs > 0)))
                    # import matplotlib.pyplot as plt
                    # plt.imshow(w_glo[:,:30], vmin=0, vmax=2)
                    # plt.colorbar()
                    # plt.show()

                        # w_glo = sess.run(model.w_glo)
                        # glo_score_w_glo, _ = tools.compute_glo_score(w_glo)
                        # log['glo_score_w_glo'].append(glo_score_w_glo, config.N_ORN)

                # Compute condition number
                # w_glo = sess.run(model.w_glo)
                # w_orn2kc = np.dot(w_orn, w_glo)
                # cond = np.linalg.cond(w_orn2kc)
                # log['cond'].append(cond)
                # print('Condition number '+ str(cond))

                if ep > 0:
                    time_spent = time.time() - start_time
                    total_time += time_spent
                    print('Time taken {:0.1f}s'.format(total_time))
                    print('Examples/second {:d}'.format(int(train_x.shape[0]/time_spent)))
                start_time = time.time()

                tools.save_log(config.save_path, log)

            if 'target_acc' in dir(config) and config.target_acc is not None:
                if res['acc'] > config.target_acc:
                    print('Training reached target accuracy {:0.2f}>{:0.2f}'.format(
                        res['acc'], config.target_acc
                    ))
                    finish_training = True

            try:
                if config.save_every_epoch and ep % config.save_epoch_interval == 0:
                    model.save_pickle(ep)
                    model.save(ep)

                # Train
                if save_everytrainloss:
                    for b in range(n_batch-1):
                        if config.separate_optimizer:
                            loss, acc, _, _ = sess.run([model.loss, model.acc, model.train_op, model.train_op1])
                        else:
                            loss, acc, _ = sess.run([model.loss, model.acc, model.train_op])
                        log['train_loss'].append(loss)

                        acc_smooth = acc_smooth * 0.9 + acc * 0.1
                        if config.target_acc is not None and acc_smooth > config.target_acc:
                            print(
                                'Training reached target accuracy {:0.2f}>{:0.2f}'.format(
                                    acc_smooth, config.target_acc
                                ))
                            finish_training = True
                            break
                else:
                    for b in range(n_batch-1):
                        _ = sess.run(model.train_op)

                        # if b % 10 == 0:
                            # w_orn, w_glo = sess.run([model.w_orn, model.w_glo])
                            # weights_over_time.append((w_orn, w_glo))

                # Compute training loss and accuracy using last batch
                loss, acc, _ = sess.run([model.loss, model.acc, model.train_op])

            except KeyboardInterrupt:
                print('Training interrupted by users')
                finish_training = True

            if finish_training:
                break

            sys.stdout.flush()

        print('Training finished')
        if 'save_log_only' in dir(config) and config.save_log_only is True:
            pass
        else:
            model.save_pickle()
            model.save()



if __name__ == '__main__':
    experiment = 'robert'
    # experiment = 'peter'
    if experiment == 'peter':
        config = SingleLayerConfig()

    elif experiment == 'robert':
        config = FullConfig()
        config.dataset = 'proto'
        config.data_dir = './datasets/proto/standard'
        config.model = 'full'
        config.save_path = './files/peter'
        config.separate_optimizer = True
        config.train_pn2kc = True
        config.sparse_pn2kc = False
    else:
        raise NotImplementedError

    train(config)


