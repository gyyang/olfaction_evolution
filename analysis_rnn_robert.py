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
import standard.analysis_pn2kc_training
import settings
import standard.analysis_activity as analysis_activity

use_torch = settings.use_torch


def _easy_weights(w_plot, x_label, y_label, dir_ix, save_path,
                  xticks=None, extra_str='', vlim=None, title=None,
                  c_label='Weight'):
    rect = [0.2, 0.15, 0.6, 0.6]
    rect_cb = [0.82, 0.15, 0.02, 0.6]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)
    if vlim == None:
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
    tools.save_fig(save_path, '__' + str(dir_ix) + '_' + y_label + '_' + x_label + '_' + extra_str, dpi=400)


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



def rnn_distribution(w_glo, dir_ix, path):
    n = os.path.join(path, '__' + str(dir_ix) + '_distribution')
    standard.analysis_pn2kc_training._plot_distribution(w_glo.flatten(), savename= n, xrange=1.0, yrange=5000)


def rnn_sparsity(w_glo, dir_ix, path):
    thres, _ = standard.analysis_pn2kc_training.infer_threshold(w_glo, visualize=False)
    claw_count = np.count_nonzero(w_glo>thres,axis=0)
    n = os.path.join(path, '__' + str(dir_ix) + '_sparsity')
    standard.analysis_pn2kc_training._plot_sparsity(claw_count, savename= n, yrange = 0.5)


def plot_activity(rnn_outputs, dir_ix, threshold, path):
    ## plot summary
    mean_activities = [np.mean(x, axis=0) for x in rnn_outputs]
    neurons_active = [np.sum(x > threshold) for x in mean_activities]
    log_neurons_active = np.log(neurons_active)
    xticks = [50, 500, 3000]


    fig = plt.figure(figsize=(1.5, 1.2 ))
    ax = fig.add_axes([0.35, 0.3, .15 * len(rnn_outputs), 0.6])
    ax.plot(log_neurons_active, 'o-', markersize=3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Active Neurons')
    ax.set_yticks(np.log(xticks))
    ax.set_yticklabels([str(x) for x in xticks])
    ax.set_xticks(np.arange(len(rnn_outputs)))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    fig_name = '__' + str(dir_ix) + '_activity'
    tools.save_fig(path, fig_name, pdf=True)


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


# def analyze_t_greater(path, dir_ix, threshold = 0.05):

if __name__ == '__main__':


    # path = './files/rnn'
    # path = './files/rnn_wdropout'
    path = './files/rnn_relabel'
    # path = './files/rnn_relabel_prune'
    # path = './files/rnn_relabel_prune2'

    from standard.analysis_activity import load_activity_torch

    # modeldir = tools.get_modeldirs(path)[0]
    # results = load_activity_torch(modeldir)
    # select_dict = {'rec_dropout_rate': 0.1, 'TIME_STEPS': 2}
    # select_dict = {'weight_dropout_rate': 0.0, 'TIME_STEPS': 2}
    # select_dict = {'diagonal': False, 'lr': 5e-4}
    select_dict = {'TIME_STEPS': 3, 'lr': 5e-4, 'diagonal': False,
                   'data_dir': './datasets/proto/relabel_200_100'}
    dir_ix = 0
# =============================================================================
#     if dir_ix == 0:
#         analyze_t0(path, dir_ix)
#     else:
#         analyze_t_greater(path, dir_ix)
# =============================================================================
    threshold = 0.05

    save_path = tools.get_modeldirs(path, select_dict=select_dict)[0]
    config = tools.load_config(save_path)
    w_rnn = tools.load_pickle(save_path)['w_rnn']  # (from, to)
    rnn_outputs = _load_activity(save_path)  # (time step, odor, neuron)

    plot_activity(rnn_outputs, dir_ix=dir_ix, path=path, threshold=threshold)

    # Find indices of neurons activated at each time point
    N_OR = config.N_ORN
    N_ORN = config.N_ORN * config.N_ORN_DUPLICATION
    ixs = []
    active_ixs = []
    assert rnn_outputs.shape[0] == config.TIME_STEPS + 1
    for i in range(config.TIME_STEPS+1):
        pn = np.mean(rnn_outputs[i], axis=0)  # average across odors
        ix = np.argsort(pn)[::-1]
        pn_cutoff= np.argmax(pn[ix] < threshold)
        pn_ix = ix[:pn_cutoff]
        ixs.append(ix)
        active_ixs.append(pn_ix)

    ## plot activity
    # ORN activation
    for t in range(rnn_outputs.shape[0]):
        ind = ixs[t]
        xticks = []

        _easy_weights(rnn_outputs[t][:100, ind], dir_ix=dir_ix,
                      y_label='Odors', x_label='Neuron',
                      xticks=xticks, title='T='+str(t),
                      save_path=path, c_label='Activity')

    ## plot weight
    w_orn = w_rnn[:N_ORN, active_ixs[1]]
    ind_max = np.argmax(w_orn, axis=1)
    ind_sort = np.argsort(ind_max)
    w_orn_reshaped = w_orn[ind_sort, :]
    _easy_weights(w_orn_reshaped.T, x_label='T=0', y_label='T=1', extra_str='reshaped', vlim=.4,
                  dir_ix=dir_ix, save_path=path)

    w_orn_mean = tools._reshape_worn(w_orn, N_OR, mode='tile')
    w_orn_mean = w_orn_mean.mean(axis=0)
    ind_max = np.argmax(w_orn_mean, axis=0)
    ind_sort = np.argsort(ind_max)
    w_orn_mean = w_orn_mean[:, ind_sort]
    _easy_weights(w_orn_mean.T, x_label='T=0', y_label='T=1', extra_str='mean',
                  dir_ix=dir_ix, save_path=path)

    w_glo = w_rnn[active_ixs[1], :]
    _easy_weights(w_glo.T, x_label='T=1', y_label='T=2',
                  dir_ix=dir_ix, save_path=path)
    rnn_distribution(w_glo, dir_ix, path)
    rnn_sparsity(w_glo, dir_ix, path)

    w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]
    
    # Plot distribution of activity
    # Third time step activity for active neurons
    figpath = tools.get_experiment_name(save_path)
    activity_threshold = 0.
    data = rnn_outputs[2][:, active_ixs[2]]
    data1 = np.mean(data > activity_threshold, axis=1)
    fname = 'spars_T3_' + tools.get_model_name(save_path)
    analysis_activity._distribution(data1, figpath, name=fname, density=False,
                  xlabel='% of Active T3 neurons',
                  ylabel='Number of Odors')

    data2 = np.mean(data > activity_threshold, axis=0)
    fname = 'spars_T3_2_' + tools.get_model_name(save_path)
    analysis_activity._distribution(data2, figpath, name=fname, density=False,
                  xlabel='% of Odors',
                  ylabel='Number of T3 neurons')
    
# =============================================================================
#     if config.TIME_STEPS == 3:
#         pn_to_pn1 = w_rnn[active_ixs[0][:,None], active_ixs[1]]
#         ind_max = np.argmax(pn_to_pn1, axis=1)
#         ind_sort = np.argsort(ind_max)
#         pn_to_pn1_reshaped = pn_to_pn1[ind_sort,:]
#         _easy_weights(pn_to_pn1_reshaped, dir_ix= dir_ix, y_label='T=1', x_label='T=2', extra_str='sorted',
#                       save_path=path)
# 
#         _easy_weights(w_glo_sorted, y_label='T=2', x_label='T=3', extra_str='sorted',
#                       dir_ix= dir_ix, save_path = path)
# 
#         w_glo_subsample = w_glo[:, 1000:1020]
#         _easy_weights(w_glo_subsample, y_label='T=2', x_label='T=3', dir_ix=dir_ix, save_path = path)
#     else:
#         _easy_weights(w_glo_sorted, y_label='T=1', x_label='T=2', extra_str='sorted',
#                       dir_ix=dir_ix, save_path=path)
# 
#         w_glo_subsample = w_glo[:, 1000:1020]
#         _easy_weights(w_glo_subsample, y_label='T=1', x_label='T=2', dir_ix=dir_ix, save_path=path)
# 
#     #EXTRAS
#     # sorted to first layer
#     for i in range(config.TIME_STEPS):
#         _easy_weights(rnn_outputs[i][:, ixs[0]], dir_ix=dir_ix, y_label='odors',
#                       x_label='Sorted to Layer 1, Layer' + '_' + str(i), save_path=path)
# 
#     # sorted to each
#     _easy_weights(rnn_outputs[0], dir_ix=dir_ix, y_label='odors', x_label='Sorted to Each, Layer_0', save_path=path)
#     for i, ix in enumerate(ixs):
#         _easy_weights(rnn_outputs[i + 1][:, ix], dir_ix=dir_ix, y_label='odors',
#                       x_label='Sorted to Each, Layer' + '_' + str(i + 1), save_path=path)
# 
#     w_orn = w_rnn[:N_ORN, active_ixs[0]]
#     w_orn_reshaped = tools._reshape_worn(w_orn, N_OR, mode='tile')
#     w_orn_reshaped = w_orn_reshaped.mean(axis=0)
#     ind_max = np.argmax(w_orn_reshaped, axis=0)
#     ind_sort = np.argsort(ind_max)
#     w_orn_reshaped = w_orn_reshaped[:, ind_sort]
#     #
#     w_glo = w_rnn[active_ixs[-1], :]
#     w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]
# 
#     if len(active_ixs) == 2:
#         pn_to_pn1 = w_rnn[active_ixs[1][:, None], active_ixs[0]]
#         ind_max = np.argmax(pn_to_pn1, axis=1)
#         ind_sort = np.argsort(ind_max)
#         pn_to_pn1_reshaped = pn_to_pn1[ind_sort, :]
#         _easy_weights(pn_to_pn1_reshaped, dir_ix=dir_ix, y_label='Layer_1', x_label='Layer_2', save_path=path)
# =============================================================================
    #
    # _easy_weights(w_rnn, y_label='Input', x_label='Output', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_rnn[:50, ixs[0]], y_label='ORN', x_label='All', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_orn, y_label='ORN', x_label='PN', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_glo, y_label='PN', x_label='KC', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_glo_sorted, y_label='PN', x_label='KC_sorted', dir_ix=dir_ix, save_path=path)
