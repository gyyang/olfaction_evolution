"""Evaluate a trained network with different amount of noise."""

import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import configs
import task
from model import FullModel
import tools
from standard.analysis import _easy_save
from tools import nicename

mpl.rcParams['font.size'] = 7

# Load dataset
data_dir = os.path.join(rootpath, 'datasets', 'proto', 'standard')
train_x, train_y, val_x, val_y = task.load_data('proto', data_dir)


def _evaluate(name, value, model, model_dir, n_rep=1):
    assert name is not 'weight_perturb'
    if model == 'oracle':
        path = os.path.join(rootpath, 'files', 'directglo_shallow', '0')
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


def _evaluate_weight_perturb(values, model, model_dir):
    if model == 'oracle':
        path = os.path.join(rootpath, 'files', 'directglo_shallow', '0')
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
        # Helper model for oracle
        oracle_x_ph = tf.placeholder(val_x.dtype,
                                     [config.N_CLASS, val_x.shape[1]])
        oracle_y_ph = tf.placeholder(val_y.dtype, [config.N_CLASS])
        oracle = FullModel(oracle_x_ph, oracle_y_ph, config=config,
                           training=False)

    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        if model == 'oracle':
            oracle.set_oracle_weights()
        else:
            val_model.load()

        val_loss = list()
        val_acc = list()
        for value in values:
            val_model.perturb_weights(value)

            # Validation
            val_loss_tmp, val_acc_tmp = sess.run(
                [val_model.loss, val_model.acc],
                {val_x_ph: val_x, val_y_ph: val_y})

            val_loss.append(val_loss_tmp)
            val_acc.append(val_acc_tmp)

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



def evaluate_weight_perturb(values, model, model_dir, n_rep=1):
    new_values = np.repeat(values, n_rep)

    val_loss, val_acc = _evaluate_weight_perturb(
        new_values, model, model_dir)

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
    loss_dict = {}
    acc_dict = {}
    for model, model_dir in zip(models, model_dirs):
        model_dir = os.path.join(path, model_dir)
        if name == 'weight_perturb':
            losses, accs = evaluate_weight_perturb(
                values, model, model_dir, n_rep=10)
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
    models = results['models']

    for ylabel in ['val_acc', 'val_loss']:
        res_dict = acc_dict if ylabel == 'val_acc' else loss_dict
        fig = plt.figure(figsize=(2.5, 2.5))
        ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
        for model in models:
            res_plot = res_dict[model]
            if ylabel == 'val_loss':
                res_plot = np.log(res_plot)
            ax.plot(values, res_plot, '-', label=model)
        plt.xlabel(nicename(name))
        plt.ylabel(nicename(ylabel))
        plt.legend()
        plt.ylim([0, 1])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ylabel == 'val_acc':
            plt.ylim([0, 1])
            ax.set_yticks([0, 0.5, 1.0])
            yrange = [0, 1]
        else:
            ax.set_yticks([-2, -1, 0, 1, 2, 3])
            yrange = [-2, 1]
        figname = ylabel + '_' + name
        # _easy_save('kc_role', figname)


def evaluate_acrossmodels():
    """Evaluate models from the same root directory."""
    name = 'weight_perturb'
    values = [0, 0.01, 0.05, 0.1, 0.3]
    n_rep = 10

    path = os.path.join(rootpath, 'files', 'vary_kc_claws_new')
    model_dirs = tools.get_allmodeldirs(path)

    loss_dict = {}
    acc_dict = {}

    models = list()
    model_var = 'kc_inputs'

    for model_dir in model_dirs:
        config = tools.load_config(model_dir)
        model = getattr(config, model_var)
        if name == 'weight_perturb':
            losses, accs = evaluate_weight_perturb(
                values, model, model_dir, n_rep=n_rep)
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

    file = os.path.join(path, name + '_' + model_var + '.pkl')
    with open(file, 'wb') as f:
        pickle.dump(results, f)


def plot_acrossmodels():
    name = 'weight_perturb'
    model_var = 'kc_inputs'

    path = os.path.join(rootpath, 'files', 'vary_kc_claws_new')
    file = os.path.join(path, name + '_' + model_var + '.pkl')
    with open(file, 'rb') as f:
        results = pickle.load(f)

    values = results['values']
    loss_dict = results['loss_dict']
    acc_dict = results['acc_dict']
    models = results['models']

    colors = plt.cm.cool(np.linspace(0, 1, len(values)))
    
    for ylabel in ['val_acc', 'val_loss']:
        res_dict = acc_dict if ylabel == 'val_acc' else loss_dict
        fig = plt.figure(figsize=(4.0, 2.5))
        ax = fig.add_axes([0.3, 0.3, 0.3, 0.6])

        for i in range(len(values)):
            res_plot = [res_dict[model][i] for model in models]
            if ylabel == 'val_loss':
                res_plot = np.log(res_plot)  # TODO: this log?
            ax.plot(models, res_plot, 'o-', label=values[i], color=colors[i])
        ax.set_xlabel(nicename(model_var))
        ax.set_ylabel(nicename(ylabel))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ylabel == 'val_acc':
            plt.ylim([0, 1])
            ax.set_yticks([0, 0.5, 1.0])
            yrange = [0, 1]
        else:
            ax.set_yticks([-2, -1, 0, 1])
            yrange = [-2, 1]
        plt.ylim(yrange)
        ax.plot([7, 7], yrange, '--', color='gray')
        l = ax.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        l.set_title(nicename(name))
        figname = ylabel+model_var+name
        _easy_save('vary_kc_claws_new', figname)
    

if __name__ == '__main__':
    # evaluate_withnoise()
    # evaluate_plot('orn_dropout_rate')
    # evaluate_plot('orn_noise_std')
    # evaluate_plot('alpha')
    path = os.path.join(rootpath, 'files', 'kcrole')
    evaluate_kcrole(path, 'weight_perturb')
    plot_kcrole(path, 'weight_perturb')
    # evaluate_acrossmodels('weight_perturb')
    # evaluate_acrossmodels()
    # plot_acrossmodels()


