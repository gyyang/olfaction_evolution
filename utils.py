import os
import numpy as np
import pickle
import tools
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from model import SingleLayerModel, FullModel
import task

def load_results(dir):
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]

    glo_score, val_acc, val_loss, train_loss, config = [], [], [], [], []
    config = []
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config.append(tools.load_config(d))
        glo_score.append(log['glo_score'])
        val_acc.append(log['val_acc'])
        val_loss.append(log['val_loss'])
        train_loss.append(log['train_loss'])
    return config, glo_score, val_acc, val_loss, train_loss

def load_pickle(dir, var):
    out = []
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    for i, d in enumerate(dirs):
        model_dir = os.path.join(d, 'model.pkl')
        with open(model_dir, 'rb') as f:
            var_dict = pickle.load(f)
            cur_val = var_dict[var]
            out.append(cur_val)
    return out

def plot_summary(data, titles, fig_dir, legends, title):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    r, c = 2, 2
    fig, ax = plt.subplots(nrows=r, ncols=c, figsize=(10, 10))
    fig.suptitle(title)
    cmap = plt.get_cmap('cool')
    colors = [cmap(i) for i in np.linspace(0, 1, len(legends))]
    for i in range(r * c):
        ax_ix = np.unravel_index(i, (r, c))
        cur_ax = ax[ax_ix]
        cur_ax.set_color_cycle(colors)
        cur_ax.plot(np.array(data[i]).transpose())
        cur_ax.legend(legends)
        cur_ax.set_xlabel('Epochs')
        cur_ax.set_ylabel(titles[i])
    plt.savefig(os.path.join(fig_dir, 'summary.png'))

def plot_summary_last_epoch(data, titles, fig_dir, parameters, title, log=True, skip=1):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    if log:
        params = np.log10(parameters)
    else:
        params = parameters
    r, c = 2, 2
    fig, ax = plt.subplots(nrows=r, ncols=c, figsize=(10, 10))
    fig.suptitle(title)
    for i, (d, t) in enumerate(zip(data, titles)):
        ax_ix = np.unravel_index(i, dims=(r, c))
        cur_ax = ax[ax_ix]
        plt.sca(cur_ax)
        y = [x[-1] for x in d]
        cur_ax.plot(params, y, marker='o')
        cur_ax.set_xlabel(title)
        cur_ax.set_ylabel(t)
        cur_ax.grid(False)
        plt.xticks(params[::skip], parameters[::skip])
        cur_ax.spines["right"].set_visible(False)
        cur_ax.spines["top"].set_visible(False)
        cur_ax.xaxis.set_ticks_position('bottom')
        cur_ax.yaxis.set_ticks_position('left')
    plt.savefig(os.path.join(fig_dir, 'summary_last_epoch.png'))


def adjust(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks([0, 0.5, 1.0])
    ax.xaxis.set_ticks([])
    ax.set_ylim([0, 1])

def plot_weights(weights, name, arg_sort, fig_dir,
                 ylabel= 'from ORNs', xlabel = 'To PNs', title='ORN-PN connectivity after training'):

    # Sort for visualization
    if arg_sort:
        ind_max = np.argmax(weights, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = weights[:, ind_sort]
        str = ''
    else:
        w_plot = weights
        str = 'unsorted_'

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect)
    vlim = .5
    im = ax.imshow(w_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim, interpolation='none')
    plt.axis('tight')
    plt.title(title)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel(xlabel, labelpad=-5)
    ax.set_ylabel(ylabel, labelpad=-5)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    plt.savefig(os.path.join(fig_dir, name + str + '_worn.png'),
                dpi=400)

def get_model_vars(save_path):
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    config.save_path = save_path

    tf.reset_default_graph()
    CurrentModel = FullModel
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    # model.save_path = save_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        # Validation
        val_loss, val_acc, glo_act, glo_in, glo_in_pre, kc_act = sess.run(
            [model.loss, model.acc, model.glo, model.glo_in, model.glo_in_pre, model.kc],
            {val_x_ph: val_x, val_y_ph: val_y})

        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    return glo_act, glo_in, glo_in_pre, kc_act