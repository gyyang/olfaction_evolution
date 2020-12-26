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

use_torch = settings.use_torch

def _easy_weights(w_plot, x_label, y_label, dir_ix, save_path, xticks=None, extra_str ='', vlim = None):
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

    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[0, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
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

    _easy_weights(rnn_outputs[0], dir_ix=dir_ix, y_label='odors', x_label='Sorted, Layer_0', save_path=path)
    _easy_weights(rnn_outputs[1], dir_ix=dir_ix, y_label='odors', x_label='Sorted, Layer_1', save_path=path)

def analyze_t_greater(path, dir_ix, threshold = 0.05):
    dirs = [os.path.join(path, n) for n in os.listdir(path)]
    save_path = dirs[dir_ix]
    config = tools.load_config(save_path)
    w_rnn = tools.load_pickles(path, 'w_rnn')[dir_ix]
    rnn_outputs = _load_activity(save_path)

    N_OR = config.N_ORN
    N_ORN = config.N_ORN * config.N_ORN_DUPLICATION
    ixs = []
    pn_ixs = []
    for i in range(1, config.TIME_STEPS):
        pn = np.mean(rnn_outputs[i], axis=0)
        ix = np.argsort(pn)[::-1]
        pn_cutoff= np.argmax(pn[ix] < threshold)
        pn_ix = ix[:pn_cutoff]
        ixs.append(ix)
        pn_ixs.append(pn_ix)
    plot_activity(rnn_outputs, dir_ix=dir_ix, path=path, threshold=threshold)

    ## plot activity
    #ORN activation
    x_range = 1000
    _easy_weights(rnn_outputs[0][:100,:x_range], dir_ix=dir_ix, y_label='Odors', x_label='T=0',
                  xticks = [0, 500, x_range],
                  extra_str='focused', save_path=path)
    _easy_weights(rnn_outputs[0][:100,:], dir_ix=dir_ix, y_label='Odors', x_label='T=0',
                  xticks = [0, 500, w_rnn.shape[0]],
                  save_path=path)
    #PN activation, sorted to T=1
    i = 1
    x_range = 100
    _easy_weights(rnn_outputs[i][:100, ixs[0][:x_range]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(i),
                  xticks=[0, 50, x_range],
                  extra_str='focused', save_path=path)
    _easy_weights(rnn_outputs[i][:100, ixs[0]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(i),
                  save_path=path)

    if config.TIME_STEPS == 3:
        i = 2
        _easy_weights(rnn_outputs[i][:100, ixs[1][:x_range]], dir_ix=dir_ix, y_label='Odors',
                      x_label='T=' + str(i),
                      xticks=[0, 50, x_range],
                      extra_str='focused', save_path=path)
        _easy_weights(rnn_outputs[i][:100, ixs[1]], dir_ix=dir_ix, y_label='Odors',
                      x_label='T=' + str(i),
                      save_path=path)

    #KC activation, sorted to T=1
    _easy_weights(rnn_outputs[config.TIME_STEPS][:100, ixs[0]], dir_ix= dir_ix, y_label='Odors',
                  x_label='T=' + str(config.TIME_STEPS),
                  save_path=path)

    ## plot weight
    w_orn = w_rnn[:N_ORN, pn_ixs[0]]
    ind_max = np.argmax(w_orn, axis=1)
    ind_sort = np.argsort(ind_max)
    w_orn_reshaped = w_orn[ind_sort,:]
    _easy_weights(w_orn_reshaped, y_label='T=0', x_label='T=1', extra_str= 'reshaped', vlim=.4,
                  dir_ix= dir_ix, save_path = path)

    w_orn_mean = tools._reshape_worn(w_orn, N_OR, mode='tile')
    w_orn_mean = w_orn_mean.mean(axis=0)
    ind_max = np.argmax(w_orn_mean, axis=0)
    ind_sort = np.argsort(ind_max)
    w_orn_mean = w_orn_mean[:, ind_sort]
    _easy_weights(w_orn_mean, y_label='T=0', x_label='T=1', extra_str='mean',
                  dir_ix= dir_ix, save_path = path)

    w_glo = w_rnn[pn_ixs[-1], :]
    rnn_distribution(w_glo, dir_ix, path)
    rnn_sparsity(w_glo, dir_ix, path)

    w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]

    if config.TIME_STEPS == 3:
        pn_to_pn1 = w_rnn[pn_ixs[0][:,None], pn_ixs[1]]
        ind_max = np.argmax(pn_to_pn1, axis=1)
        ind_sort = np.argsort(ind_max)
        pn_to_pn1_reshaped = pn_to_pn1[ind_sort,:]
        _easy_weights(pn_to_pn1_reshaped, dir_ix= dir_ix, y_label='T=1', x_label='T=2', extra_str='sorted',
                      save_path=path)

        _easy_weights(w_glo_sorted, y_label='T=2', x_label='T=3', extra_str='sorted',
                      dir_ix= dir_ix, save_path = path)

        w_glo_subsample = w_glo[:, 1000:1020]
        _easy_weights(w_glo_subsample, y_label='T=2', x_label='T=3', dir_ix=dir_ix, save_path = path)
    else:
        _easy_weights(w_glo_sorted, y_label='T=1', x_label='T=2', extra_str='sorted',
                      dir_ix=dir_ix, save_path=path)

        w_glo_subsample = w_glo[:, 1000:1020]
        _easy_weights(w_glo_subsample, y_label='T=1', x_label='T=2', dir_ix=dir_ix, save_path=path)

    #EXTRAS
    # sorted to first layer
    for i in range(config.TIME_STEPS):
        _easy_weights(rnn_outputs[i][:, ixs[0]], dir_ix=dir_ix, y_label='odors',
                      x_label='Sorted to Layer 1, Layer' + '_' + str(i), save_path=path)

    # sorted to each
    _easy_weights(rnn_outputs[0], dir_ix=dir_ix, y_label='odors', x_label='Sorted to Each, Layer_0', save_path=path)
    for i, ix in enumerate(ixs):
        _easy_weights(rnn_outputs[i + 1][:, ix], dir_ix=dir_ix, y_label='odors',
                      x_label='Sorted to Each, Layer' + '_' + str(i + 1), save_path=path)

    w_orn = w_rnn[:N_ORN, pn_ixs[0]]
    w_orn_reshaped = tools._reshape_worn(w_orn, N_OR, mode='tile')
    w_orn_reshaped = w_orn_reshaped.mean(axis=0)
    ind_max = np.argmax(w_orn_reshaped, axis=0)
    ind_sort = np.argsort(ind_max)
    w_orn_reshaped = w_orn_reshaped[:, ind_sort]
    #
    w_glo = w_rnn[pn_ixs[-1], :]
    w_glo_sorted = np.sort(w_glo, axis=0)[::-1, :]

    if len(pn_ixs) == 2:
        pn_to_pn1 = w_rnn[pn_ixs[1][:, None], pn_ixs[0]]
        ind_max = np.argmax(pn_to_pn1, axis=1)
        ind_sort = np.argsort(ind_max)
        pn_to_pn1_reshaped = pn_to_pn1[ind_sort, :]
        _easy_weights(pn_to_pn1_reshaped, dir_ix=dir_ix, y_label='Layer_1', x_label='Layer_2', save_path=path)
    #
    # _easy_weights(w_rnn, y_label='Input', x_label='Output', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_rnn[:50, ixs[0]], y_label='ORN', x_label='All', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_orn, y_label='ORN', x_label='PN', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_glo, y_label='PN', x_label='KC', dir_ix=dir_ix, save_path=path)
    # _easy_weights(w_glo_sorted, y_label='PN', x_label='KC_sorted', dir_ix=dir_ix, save_path=path)

if __name__ == '__main__':


    path = '../files/rnn'

    from standard.analysis_activity import load_activity_torch

    modeldir = tools.get_modeldirs(path)[0]
    results = load_activity_torch(modeldir)
    # dir_ix = 1
    # if dir_ix == 0:
    #     analyze_t0(path, dir_ix)
    # else:
    #     analyze_t_greater(path, dir_ix)