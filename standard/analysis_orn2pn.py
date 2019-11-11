import numpy as np
import os

from matplotlib import pyplot as plt

import standard.analysis as sa
import tools
import matplotlib.pyplot as plt
import task
import tensorflow as tf
from model import FullModel
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'







def _correlation(mat):
    corrcoef = np.corrcoef(mat, rowvar=False)
    mask = ~np.eye(corrcoef.shape[0], dtype=bool)
    nanmask = ~np.isnan(corrcoef)
    flattened_corrcoef = corrcoef[np.logical_and(mask, nanmask)]
    return np.mean(flattened_corrcoef)

def correlation_across_epochs(save_path, legend):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    ys = []
    for i, d in enumerate(dirs):
        list_of_corr_coef = []
        dir_with_epoch = os.path.join(d, 'epoch')
        epoch_dirs = [os.path.join(dir_with_epoch, x) for x in os.listdir(dir_with_epoch)]
        for epoch_dir in epoch_dirs:
            glo_in, glo_out, kc_out, results = _load_epoch_activity(d, epoch_dir)
            list_of_corr_coef.append(_correlation(glo_out))
        ys.append(list_of_corr_coef)
    _plot_progress(ys, legend, save_path, '_correlation_progress',
                   ylim = [-0.05, 1], yticks = [0, 0.5, 1.0], ylabel= 'Correlation')

def _plot_progress(ys, legend, save_path, name, ylim, yticks, ylabel):
    y = ys[0]
    figsize = (1.5, 1.2)
    rect = [0.3, 0.3, 0.65, 0.5]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    xlim = len(y)
    ax.plot(np.transpose(ys))
    xticks = np.arange(0, xlim, 5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim([0, len(y) - 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend(legend, fontsize=4, frameon=False)

    from tools import save_fig
    save_fig(save_path, name, dpi=500)

def _load_epoch_activity(config_path, epoch_path):
    '''
    Loads model activity from tensorflow
    :param config_path:
    :return:
    '''

    # # Reload the network and analyze activity
    config = tools.load_config(config_path)
    config.data_dir = config.data_dir #hack
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    tf.reset_default_graph()
    CurrentModel = FullModel

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    model.save_path = epoch_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        # Validation
        glo_out, glo_in, kc_out, logits = sess.run(
            [model.glo, model.glo_in, model.kc, model.logits],
            {val_x_ph: val_x, val_y_ph: val_y})
        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    return glo_in, glo_out, kc_out, results


def _plot_gloscores(path, ix, cutoff, shuffle=False, vlim=[0, 5]):
    def _helper_mat(w_plot, string):
        rect = [0.15, 0.15, 0.65, 0.65]
        rect_cb = [0.82, 0.15, 0.02, 0.65]
        fig = plt.figure(figsize=(2.6, 2.6))
        ax = fig.add_axes(rect)
        cmap = tools.get_colormap()

        ind_max = np.argmax(w_plot, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_plot[:, ind_sort]

        im = ax.imshow(w_plot, cmap=cmap, vmin=vlim[0], vmax=vlim[1], interpolation='none')

        plt.axis('tight')
        for loc in ['bottom', 'top', 'left', 'right']:
            ax.spines[loc].set_visible(False)
        ax.tick_params('both', length=0)
        ax.set_xticks([0, w_plot.shape[1]])
        ax.set_yticks([0, w_plot.shape[0]])
        ax.set_xlabel('To PN')
        ax.set_ylabel('From ORN')
        ax.set_title(string)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vlim[0], vlim[1]])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Weight', fontsize=7, labelpad=-10)
        plt.tick_params(axis='both', which='major', labelsize=7)
        plt.axis('tight')

        tools.save_fig(path, 'ix_' + str(ix) + '_cutoff_' + string)

    w_orns = tools.load_pickle(path, 'w_orn')
    w_orn = w_orns[ix]
    w_orn = tools._reshape_worn(w_orn, unique_orn=50)
    w_orn = w_orn.mean(axis=0)
    print(w_orn.shape)

    avg_gs, all_gs = tools.compute_glo_score(w_orn, unique_ors=50, mode='tile', w_or=None)
    all_gs = np.array(all_gs)
    ix_good = all_gs >= cutoff
    ix_bad = all_gs < cutoff

    arg = 'shuffled' if shuffle else ''
    _helper_mat(w_orn[:, ix_good], 'top_' + arg)
    _helper_mat(w_orn[:, ix_bad], 'bottom_' + arg)

    if shuffle:
        np.random.shuffle(w_orn.flat)
        all_gs = []
        for i in range(10):
            np.random.shuffle(w_orn.flat)
            avg_gs, all_gs_ = tools.compute_glo_score(w_orn, 50, mode='tile', w_or=None)
            all_gs.append(all_gs_)
        all_gs = np.concatenate(all_gs, axis=0)

    arg = '_ix_' + str(ix) + '_hist_'
    arg = arg + 'shuffled' if shuffle else arg
    plt.figure()

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(all_gs, bins=50, range=[0, 1], align='left')
    plt.plot([cutoff, cutoff], [0, ax.get_ylim()[1]], '--', color='gray')
    ax.set_xlabel('Glo Score')
    ax.set_ylabel('Count')
    ax.set_title('GloScore Distribution For all PNs')

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlim([-0.05, 1.05])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    tools.save_fig(path, arg)
    return ix_good, ix_bad


def _distribution_multiglomerular_pn(path, ix, ix_good, ix_bad):
    # weights
    w_kcs = tools.load_pickle(path, 'w_glo')
    w_kc = w_kcs[ix]
    sum_kc_weights = np.sum(w_kc, axis=1)
    weight_to_good = sum_kc_weights[ix_good]
    weight_to_bad = sum_kc_weights[ix_bad]

    arg = '_ix_' + str(ix) + '_pn2kc_hist_'
    plt.figure()

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    alpha = 0.8
    plt.hist(weight_to_good, bins=100, range=[0, 300], alpha = alpha)
    plt.hist(weight_to_bad, bins=100, range=[0, 300], alpha = alpha)
    ax.set_xlabel('PN-KC Weights')
    ax.set_ylabel('Count')
    plt.legend(['UniGlo', 'MultiGlo'], frameon=False, loc=1)

    # ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    # plt.xlim([-0.05, 1.05])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    tools.save_fig(path, arg)

def _lesion_multiglomerular_pn(path, units):
    import tensorflow as tf
    from model import FullModel

    config = tools.load_config(path)
    tf.reset_default_graph()

    # Load dataset
    config.dataset = 'proto'
    config.data_dir = './datasets/proto/standard'
    config.save_path = path
    train_x, train_y, val_x, val_y = task.load_data(
        config.dataset, config.data_dir)

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_model.load()

        if units is not None:
            val_model.lesion_units('model/layer2/kernel:0', units)

        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})
        print(val_acc)
        return val_acc

def lesion_glom(path, ix, ix_good, ix_bad):
    acc0 = _lesion_multiglomerular_pn(os.path.join(path,'0000' + str(ix)), None)
    acc1 = _lesion_multiglomerular_pn(os.path.join(path,'0000' + str(ix)), ix_good)
    acc2 = _lesion_multiglomerular_pn(os.path.join(path,'0000' + str(ix)), ix_bad)
    bars = [acc0, acc1, acc2]

    fs = 6
    width = 0.7
    ylim = [0, 1]
    fig = plt.figure(figsize=(1.2, 1.2))
    ax = fig.add_axes([0.35, 0.35, 0.6, 0.4])
    xlocs = np.arange(len(bars))
    b0 = ax.bar(xlocs, bars, width=width, edgecolor='none')
    ax.set_xticks(xlocs)
    ax.set_xticklabels(['None', 'UniGlo', 'MultiGlo'], rotation=25)
    ax.set_xlabel('Lesioning', labelpad=-2)
    ax.set_ylabel('Accuracy')
    ax.tick_params(axis='both', which='major')
    plt.locator_params(axis='y', nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.set_xlim([-0.8, len(rules_perf)-0.2])
    ax.set_ylim(ylim)
    ax.set_yticks(ylim)
    tools.save_fig(path, str = '_' + str(ix) + '_lesion')

if __name__ == '__main__':
    path = '../files/control_vary_pn'
    ix = 22

    ix_good, ix_bad = _plot_gloscores(path, ix, cutoff=.9, shuffle=False)
    _distribution_multiglomerular_pn(path, ix, ix_good, ix_bad)

    print(np.sum(ix_good))
    print(np.sum(ix_bad))
    lesion_glom(path, ix, ix_good, ix_bad)
