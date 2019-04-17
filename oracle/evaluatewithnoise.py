"""Evaluate a trained network with different amount of noise."""

import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import configs
import task
from model import FullModel
import tools
from standard.analysis import _easy_save
from tools import nicename

mpl.rcParams['font.size'] = 7

oracle_dir = 'kcrole'
# Load dataset
# TODO: Make sure this works for dataset is different
data_dir = os.path.join(rootpath, 'datasets', 'proto', 'small')
# data_dir = os.path.join(rootpath, 'datasets', 'proto', 'standard')
train_x, train_y, val_x, val_y = task.load_data('proto', data_dir)


def _evaluate(name, value, model, model_dir, n_rep=1):
    assert name is not 'weight_perturb'
    if model == 'oracle':
        path = os.path.join(rootpath, 'files', oracle_dir, '000000')
    else:
        path = model_dir

    config = tools.load_config(path)

    # TODO: clean up these paths
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    if name == 'orn_noise_std':
        config.ORN_NOISE_STD = value
    elif name == 'orn_dropout_rate':
        config.orn_dropout = True
        config.orn_dropout_rate = value
    elif name == 'alpha':
        config.oracle_scale = value
    else:
        raise ValueError('Unknown name', str(name))

    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)

    if model == 'oracle':
        # Over-write config
        config.skip_orn2pn = True
        config.skip_pn2kc = True
        config.kc_dropout = False
        config.set_oracle = True
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype, [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = FullModel(oracle_x_ph, oracle_y_ph, config=config, training=False)
    
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if model == 'oracle':
            oracle.set_oracle_weights()
        else:
            val_model.load()

        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})

    return val_loss, val_acc


def perturb_weights(self, scale, perturb_var=None, perturb_dir=None):
    """Perturb all weights with multiplicative noise.

    Args:
        scale: float. Perturb weights with
            random variables ~ U[1-scale, 1+scale]
        perturb_var: list of variables names to be perturbed
        perturb_dir: if not None, the direction of weight perturbation.
            By default, the weights
    """
    sess = tf.get_default_session()



    def perturb(w, d):
        w = w + d * scale
        w = w * np.random.uniform(1-scale, 1+scale, size=w.shape)
        return w

    # record original weight values when perturb for the first time
    if not hasattr(self, 'origin_weights'):
        print('Perturbing weights:')
        for v in perturb_var:
            print(v)
        self.origin_weights = [sess.run(v) for v in perturb_var]

    for v_value, v in zip(self.origin_weights, perturb_var):
        sess.run(v.assign(perturb(v_value)))


def _evaluate_weight_perturb(values, model, model_dir, dataset='val',
                             perturb_mode='multiplicative'):
    if model == 'oracle':
        path = os.path.join(rootpath, 'files', oracle_dir, '000000')
    else:
        path = model_dir

    config = tools.load_config(path)

    # TODO: clean up these paths
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)

    if model == 'oracle':
        # Over-write config
        config.skip_orn2pn = True
        config.skip_pn2kc = True
        config.kc_dropout = False
        config.set_oracle = True
        config.oracle_scale = 8
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype,
                                     [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = FullModel(oracle_x_ph, oracle_y_ph, config=config,
                           training=False)

    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)

    # Variables to perturb
    perturb_var = None
    perturb_var = ['model/layer3/kernel:0']

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if model == 'oracle':
            oracle.set_oracle_weights()
        else:
            val_model.load()

        if dataset == 'val':
            data_x, data_y = val_x, val_y
        elif dataset == 'train':
            rnd_ind = np.random.choice(
                train_x.shape[0], size=(val_x.shape[0],), replace=False)
            data_x, data_y = train_x[rnd_ind], train_y[rnd_ind]
        else:
            raise ValueError('Wrong dataset type')

        print('Perturbing weights:')
        for v in perturb_var:
            print(v)

        if perturb_var is None:
            perturb_var = tf.trainable_variables()
        else:
            perturb_var = [v for v in tf.trainable_variables() if
                           v.name in perturb_var]

        origin_weights = [sess.run(v) for v in perturb_var]

        val_loss = list()
        val_acc = list()
        reps = 10

        for value in values:
            temp_loss, temp_acc = 0, 0
            for rep in range(reps):

                if perturb_mode == 'multiplicative':
                    new_var_val = [w*np.random.uniform(1-value, 1+value, size=w.shape)
                                   for w in origin_weights]
                else:
                    raise ValueError()

                for j in range(len(perturb_var)):
                    sess.run(perturb_var[j].assign(new_var_val[j]))

                # Validation
                val_loss_tmp, val_acc_tmp = sess.run(
                    [val_model.loss, val_model.acc],
                    {val_x_ph: data_x, val_y_ph: data_y})
                temp_loss += val_loss_tmp
                temp_acc += val_acc_tmp
            temp_loss /= reps
            temp_acc /= reps

            val_loss.append(temp_loss)
            val_acc.append(temp_acc)

    return val_loss, val_acc


def evaluate(name, values, model, model_dir, n_rep=1):
    losses = list()
    accs = list()
    for value in values:
        val_loss, val_acc = _evaluate(name, value, model, model_dir,
                                      n_rep=n_rep)
        losses.append(val_loss)
        accs.append(val_acc)
    return losses, accs


def evaluate_weight_perturb(values, model, model_dir, n_rep=1, dataset='val'):
    new_values = np.repeat(values, n_rep)

    val_loss, val_acc = _evaluate_weight_perturb(
        new_values, model, model_dir, dataset=dataset)

    losses = np.array(val_loss).reshape(len(values), n_rep).mean(axis=1)
    accs = np.array(val_acc).reshape(len(values), n_rep).mean(axis=1)
    return losses, accs


def evaluate_kcrole(path, name):
    """Evaluate KC layer's role."""
    if name == 'orn_dropout_rate':
        values = np.linspace(0, 0.3, 10)
    elif name == 'orn_noise_std':
        values = np.linspace(0, 0.3, 10)
    elif name == 'alpha':
        values = np.linspace(0.2, 8, 10)
    elif name == 'weight_perturb':
        values = [0, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    else:
        raise ValueError()

    models = ['oracle', 'sgd + no kc', 'sgd + trained kc', 'sgd + fixed kc']
    model_dirs = ['none', '000002', '000000', '000001']
    # models = ['oracle', 'sgd + no kc', 'sgd + trained kc']
    # model_dirs = ['none', '000002', '000000']
    loss_dict = {}
    acc_dict = {}
    for model, model_dir in zip(models, model_dirs):
        model_dir = os.path.join(path, model_dir)
        if name == 'weight_perturb':
            losses, accs = evaluate_weight_perturb(
                values, model, model_dir, n_rep=1)
        else:
            losses, accs = evaluate(name, values, model, model_dir)
        loss_dict[model] = losses
        acc_dict[model] = accs

    results = {'loss_dict': loss_dict,
               'acc_dict': acc_dict,
               'models': models,
               'values': values,
               'name': name}

    file = os.path.join(path, 'vary_' + name + '.pkl')
    with open(file, 'wb') as f:
        pickle.dump(results, f)


def plot_kcrole(path, name):
    file = os.path.join(path, 'vary_' + name + '.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)

    values = results['values']
    loss_dict = results['loss_dict']
    acc_dict = results['acc_dict']
    # models = results['models']
    models = ['oracle', 'sgd + no kc', 'sgd + trained kc']
    for ylabel in ['val_acc', 'val_loss']:
        res_dict = acc_dict if ylabel == 'val_acc' else loss_dict
        fig = plt.figure(figsize=(2.75, 2))
        ax = fig.add_axes([0.2, 0.2, 0.4, 0.6])
        for model in models:
            res_plot = res_dict[model]
            if ylabel == 'val_loss':
                res_plot = np.log(res_plot)
            ax.plot(values, res_plot, 'o-', markersize=3, label=model)
        plt.xlabel(nicename(name))
        plt.ylabel(nicename(ylabel))
        plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0), frameon=False)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ylabel == 'val_acc':
            plt.ylim([0, 1.05])
            ax.set_yticks([0, 0.5, 1.0])
        else:
            ax.set_yticks([-2, -1, 0, 1, 2, 3, 4])
            plt.ylim([-2, 4])
        figname = ylabel + '_' + name
        _easy_save('kc_role', figname)


def select_config(config, select_dict):
    if select_dict is not None:
        for key, val in select_dict.items():
            if getattr(config, key) != val:
                return False
    return True


def evaluate_acrossmodels(path, select_dict=None, dataset='val'):
    """Evaluate models from the same root directory."""
    name = 'weight_perturb'
    # values = [0, 0.05, 0.1]
    values = [0, 0.2, 0.5]
    # n_rep = 10
    n_rep = 1

    # path = os.path.join(rootpath, 'files', 'vary_kc_claws_new')
    # path = os.path.join(rootpath, 'files', 'tmp_perturb_nobias')
    model_dirs = tools.get_allmodeldirs(path)

    loss_dict = {}
    acc_dict = {}

    models = list()
    model_var = 'kc_inputs'

    for model_dir in model_dirs:
        config = tools.load_config(model_dir)
        if not select_config(config, select_dict):
            continue
        
        model = getattr(config, model_var)
        if name == 'weight_perturb':
            losses, accs = evaluate_weight_perturb(
                values, model, model_dir, n_rep=n_rep, dataset=dataset)
        else:
            losses, accs = evaluate(name, values, model, model_dir, n_rep=n_rep)
        loss_dict[model] = losses
        acc_dict[model] = accs
        models.append(model)

    results = {'loss_dict': loss_dict,
               'acc_dict': acc_dict,
               'models': models,
               'model_var': model_var,
               'values': values,
               'name': name}

    file = os.path.join(path, name + '_' + model_var+ '_' + dataset + '.pkl')
    with open(file, 'wb') as f:
        pickle.dump(results, f)


def plot_acrossmodels(path, dataset='val'):
    name = 'weight_perturb'
    model_var = 'kc_inputs'

    # path = os.path.join(rootpath, 'files', 'vary_kc_claws_new')
    # path = os.path.join(rootpath, 'files', 'tmp_perturb_nobias')
    file = os.path.join(path, name + '_' + model_var+ '_' + dataset + '.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)

    values = results['values']
    loss_dict = results['loss_dict']
    acc_dict = results['acc_dict']
    models = results['models']
    print(models)

    colors = plt.cm.cool(np.linspace(0, 1, len(values)))
    
    for ylabel in ['val_acc', 'val_loss']:
        res_dict = acc_dict if ylabel == 'val_acc' else loss_dict
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_axes([0.2, 0.25, 0.45, 0.5])

        for i in range(len(values)):
            res_plot = [res_dict[model][i] for model in models]
            if ylabel == 'val_logloss':
                res_plot = np.log(res_plot)  # TODO: this log?
            print(res_plot)
            ax.plot(models, res_plot, 'o-', markersize=3, label=values[i], color=colors[i])
        ax.set_xlabel(nicename(model_var))
        ax.set_ylabel(nicename(ylabel))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
# =============================================================================
#         if ylabel == 'val_acc':
#             ax.set_yticks([0.8, 0.9, 1.0])
#             yrange = [0.8, 1]
#         else:
#             ax.set_yticks([-2, -1, 0, 1])
#             yrange = [-2.5, -0.5]
#         ax.set_xticks([3, 7, 15, 30, 50])
#         plt.ylim(yrange)
# =============================================================================
        # ax.plot([7, 7], yrange, '--', color='gray')
        l = ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0), frameon = False)
        l.set_title(nicename(name))
        figname = ylabel+model_var+name+dataset
        _easy_save(path.split('/')[-1], figname)


if __name__ == '__main__':
    # evaluate_withnoise()
    # evaluate_plot('orn_dropout_rate')
    # evaluate_plot('orn_noise_std')
    # evaluate_plot('alpha')
    # path = os.path.join(rootpath, 'files', 'kcrole')
    # evaluate_kcrole(path, 'weight_perturb')
    # plot_kcrole(path, 'weight_perturb')
    # evaluate_acrossmodels('weight_perturb')
    path = os.path.join(rootpath, 'files', 'tmp_perturb_small')
    evaluate_acrossmodels(path, select_dict={'ORN_NOISE_STD': 0}, dataset='train')
    plot_acrossmodels(path, dataset='train')
    
# =============================================================================
#     from standard.analysis import plot_progress
#     path = os.path.join(rootpath, 'files', 'tmp_perturb_nobias')
#     plot_progress(path, exclude_epoch0=True)
# =============================================================================


