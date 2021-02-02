#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 26 16:57:49 2020

@author: gryang
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import task
import tools
import standard.analysis as sa
import standard.analysis_pn2kc_training as analysis_pn2kc_training
import settings
import standard.analysis_activity as analysis_activity

use_torch = settings.use_torch


def _easy_weights(w_plot, modeldir, x_label=None, y_label=None,
                  xticks=None, extra_str='', vlim=None, title=None,
                  c_label='Weight'):
    rect = [0.15, 0.15, 0.6, 0.6]
    rect_cb = [0.77, 0.15, 0.02, 0.6]
    fig = plt.figure(figsize=(1.7, 1.7))
    ax = fig.add_axes(rect)
    if vlim is None:
        vlim = np.round(np.max(abs(w_plot)), decimals=1)

    cmap = plt.get_cmap('RdBu_r')
    positive_cmap = np.min(w_plot) > -1e-6  # all weights positive
    if positive_cmap:
        cmap = tools.truncate_colormap(cmap, 0.5, 1.0)

    im = ax.imshow(w_plot, cmap=cmap, vmin=0, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xticks:
        ax.set_xticks(xticks)
    else:
        ax.set_xticks([0, w_plot.shape[1]])

    if title is not None:
        ax.set_title(title, fontsize=7)

    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[0, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label(c_label, fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    tools.save_fig(tools.get_experiment_name(modeldir),
                   '_' + tools.get_model_name(modeldir) +
                   '_' + y_label + '_' + x_label + '_' + extra_str, dpi=400)


def _load_activity_tf(save_path):
    import tensorflow as tf
    import model as network_models

    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    config.data_dir = r'C:\Users\Peter\PycharmProjects\olfaction_evolution\datasets\proto\orn50'
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

    tf.reset_default_graph()
    CurrentModel = network_models.RNN
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    model.save_path = save_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()
        rnn_outputs = sess.run(model.rnn_outputs, {val_x_ph: val_x, val_y_ph: val_y})
    return rnn_outputs


def _load_activity(save_path):
    if use_torch:
        from standard.analysis_activity import load_activity_torch
        results = load_activity_torch(save_path)
        return results['rnn_outputs']
    else:
        return _load_activity_tf(save_path)


def rnn_distribution(w_glo, modeldir):
    """Plot distribution of effective PN-KC weights."""
    savename = os.path.join(
        tools.get_experiment_name(modeldir),
        '_' + tools.get_model_name(modeldir) + '_distribution')
    standard.analysis_pn2kc_training._plot_distribution(
        w_glo.flatten(), savename=savename, xrange=1.0, yrange=5000)
    standard.analysis_pn2kc_training._plot_log_distribution(
        w_glo.flatten(), savename=savename, res_fit=True)


def rnn_sparsity(w_glo, modeldir):
    """Plot sparsity of effective PN-KC connections."""
    thres, _ = standard.analysis_pn2kc_training.infer_threshold(
        w_glo, visualize=False)
    claw_count = np.count_nonzero(w_glo > thres, axis=0)
    n = os.path.join(tools.get_experiment_name(modeldir),
                     '_' + tools.get_model_name(modeldir) + '_sparsity')
    standard.analysis_pn2kc_training._plot_sparsity(
        claw_count, savename=n, yrange=0.5)


def plot_num_active_neurons(active_ixs, modeldir):
    """Plot activity of RNN.
    
    Args:
        active_ixs: list of arrays
    """
    neurons_active = [len(ind) for ind in active_ixs]
    log_neurons_active = np.log(neurons_active)
    xticks = [50, 500, 2500]

    fig = plt.figure(figsize=(1.5, 1.2))
    ax = fig.add_axes([0.35, 0.3, .5, 0.6])
    ax.plot(log_neurons_active, 'o-', markersize=3)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Active Neurons')
    ax.set_yticks(np.log(xticks))
    ax.set_yticklabels([str(x) for x in xticks])
    ax.set_xticks(np.arange(len(neurons_active)))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.grid(axis='y')

    fig_name = 'num_neurons_by_step_' + tools.get_model_name(modeldir)
    tools.save_fig(tools.get_experiment_name(modeldir), fig_name, pdf=True)


def _plot_all_activity(rnn_activity, modeldir):
    """Plot activity for all odors and time steps.

    Args:
        rnn_activity: (Step, Batch, Neuron)
    """
    for step in range(rnn_activity.shape[0]):
        mean_activity = np.mean(rnn_activity[step], axis=0)  # (Neuron)
        ind_sort = np.argsort(mean_activity)[::-1]
        _easy_weights(rnn_activity[step, :100][:, ind_sort],
                      modeldir=modeldir, y_label='Odors', x_label='Neuron',
                      title='Step='+str(step), c_label='Activity')


def analyze_rnn_activity(modeldir, threshold=0.15, plot=True):
    """Analyze activity of RNN.

    Returns:
        rnn_activity: (Step, Batch, Neuron)
        active_ixs: list of arrays
    """
    # Find indices of neurons activated at each time point
    config = tools.load_config(modeldir)

    rnn_activity = _load_activity(modeldir)  # (Step, Batch, Neuron)
    assert rnn_activity.shape[0] == config.TIME_STEPS + 1

    # Compute the number of active neurons at each step
    mean_activity = np.mean(rnn_activity, axis=1)  # (Step, Neuron)

    ixs = []
    active_ixs = []
    for ma in mean_activity:
        ind_sort = np.argsort(ma)[::-1]
        ixs.append(ind_sort)
        # Threshold may not be robust
        active_ixs.append(np.where(ma > threshold)[0])

    # Store weights based on active indices
    res = tools.load_pickle(modeldir)  # (from, to)
    res.allow_pickle = True
    res = dict(res)
    w_rnn = res['w_rnn']
    N_ORN = config.N_PN * config.N_ORN_DUPLICATION
    w_orn = w_rnn[:N_ORN, active_ixs[1]]
    new_res = {'w_orn': w_orn, 'active_ixs': active_ixs}
    if config.TIME_STEPS == 2:
        w_glo = w_rnn[active_ixs[1], :][:, active_ixs[2]]
        new_res['w_glo'] = w_glo
    if config.TIME_STEPS == 3:
        w_step2to3 = w_rnn[active_ixs[1], :][:, active_ixs[2]]
        w_step3to4 = w_rnn[active_ixs[2], :][:, active_ixs[3]]
        new_res['w_glo'] = np.dot(w_step2to3, w_step3to4)
        new_res['w_step2to3'] = w_step2to3
        new_res['w_step3to4'] = w_step3to4

    res.update(new_res)
    tools.save_pickle(modeldir, res)

    if plot:
        # Plotting results
        _plot_all_activity(rnn_activity, modeldir)
        plot_num_active_neurons(active_ixs, modeldir)

    return rnn_activity, active_ixs


def analyze_rnn_weights(modeldir):
    config = tools.load_config(modeldir)
    sa.plot_weights(modeldir, 'w_orn',
                    ax_args={'xlabel': 'From Step 1', 
                             'ylabel': 'To Step 2',
                             'title': 'Step 1-2 connectivity'})

    # Only plotted if exists
    sa.plot_weights(modeldir, 'w_step2to3',
                    ax_args={'xlabel': 'From Step 2', 
                             'ylabel': 'To Step 3',
                             'title': 'Step 2-3 connectivity'})

    sa.plot_weights(modeldir, 'w_step3to4',
                    ax_args={'xlabel': 'From Step 3',
                             'ylabel': 'To Step 4',
                             'title': 'Step 3-4 connectivity'})

    n_step = config.TIME_STEPS
    if n_step == 2:
        title = 'Step 2-3 connectivity'
    else:
        title = 'Effective Step 2-{:d} connectivity'.format(n_step+1)
    sa.plot_weights(
        modeldir, 'w_glo',
        ax_args={'xlabel': 'From Step 2',
                 'ylabel': 'To Step ' + str(n_step + 1),
                 'title': title})
    analysis_pn2kc_training.plot_distribution(modeldir)
    analysis_pn2kc_training.plot_sparsity(modeldir)


def analyze_t0(path, dir_ix):
    dirs = [os.path.join(path, n) for n in os.listdir(path)]
    save_path = dirs[dir_ix]
    config = tools.load_config(save_path)
    w_rnn = tools.load_pickles(path, 'w_rnn')[dir_ix]
    rnn_outputs = _load_activity(save_path)

    N_ORN = config.N_ORN * config.N_ORN_DUPLICATION
    w_glo = w_rnn[:N_ORN, N_ORN:]
    w_glo_subsample = w_glo[:, :20]
    _easy_weights(w_glo_subsample, y_label='T=0', x_label='T=1', dir_ix=dir_ix, save_path = path)
    rnn_distribution(w_glo, dir_ix, path)
    rnn_sparsity(w_glo, dir_ix, path)

    _easy_weights(rnn_outputs[0], dir_ix=dir_ix, y_label='odors', 
                  x_label='Sorted, Layer_0', save_path=path)
    _easy_weights(rnn_outputs[1], dir_ix=dir_ix, y_label='odors', 
                  x_label='Sorted, Layer_1', save_path=path)



if __name__ == '__main__':


    # path = './files/rnn'
    # path = './files/rnn_wdropout'
    path = './files/rnn_relabel'
    # path = './files/rnn_relabel_noreactivation'
    # path = './files/rnn_relabel_prune'
    # path = './files/rnn_relabel_prune'

    select_dict = {'TIME_STEPS': 3, 'lr': 2e-4, 'diagonal': False,
                   'data_dir': './datasets/proto/relabel_200_100'}

    modeldir = tools.get_modeldirs(path, select_dict=select_dict)[0]

    analyze_rnn_activity(modeldir)
    analyze_rnn_weights(modeldir)
