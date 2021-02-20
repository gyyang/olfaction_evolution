import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

import tools
import task


mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def correlation_across_epochs(save_path, legend = None, arg = 'weight'):
    import tensorflow as tf
    from model import FullModel

    def _correlation(mat):
        corrcoef = np.corrcoef(mat, rowvar=False)
        mask = ~np.eye(corrcoef.shape[0], dtype=bool)
        nanmask = ~np.isnan(corrcoef)
        flattened_corrcoef = corrcoef[np.logical_and(mask, nanmask)]
        return np.mean(flattened_corrcoef)

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
        if legend is not None:
            plt.legend(legend, fontsize=7, frameon=False)

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
        config.data_dir = config.data_dir  # hack
        train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

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

    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    ys = []
    for i, d in enumerate(dirs):
        list_of_corr_coef = []
        epoch_dirs = tools.get_modeldirs(os.path.join(d,'epoch'))
        for epoch_dir in epoch_dirs:
            if arg == 'weight':
                data = tools.load_pickles(epoch_dir, 'w_orn')[0]
            elif arg == 'activity':
                glo_in, glo_out, kc_out, results = _load_epoch_activity(d, epoch_dir)
                data = glo_out
            else:
                raise ValueError('argument is unrecognized'.format(arg))
            list_of_corr_coef.append(_correlation(data))
        ys.append(list_of_corr_coef)
    _plot_progress(ys, legend, save_path, '_correlation_' + str(arg),
                   ylim = [-0.2, 1], yticks = [0, 0.5, 1.0], ylabel= 'Correlation')


def multiglo_gloscores(modeldir, cutoff, shuffle=False, vlim=[0, 5]):
    def _helper_mat(w_plot, string):
        figsize = (1.7, 1.7)
        rect = [0.15, 0.15, 0.6, 0.6]
        rect_cb = [0.77, 0.15, 0.02, 0.6]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)
        # cmap = tools.get_colormap()
        cmap = plt.get_cmap('RdBu_r')
        positive_cmap = np.min(w_plot) > -1e-6  # all weights positive
        if positive_cmap:
            cmap = tools.truncate_colormap(cmap, 0.5, 1.0)

        ind_max = np.argmax(w_plot, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_plot[:, ind_sort]

        im = ax.imshow(w_plot.T, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                       interpolation='none')

        plt.axis('tight')
        for loc in ['bottom', 'top', 'left', 'right']:
            ax.spines[loc].set_visible(False)
        ax.tick_params('both', length=0)
        ax.set_yticks([0, w_plot.shape[1]])
        ax.set_xticks([0, w_plot.shape[0]])
        labelpad = -5
        ax.set_ylabel('To PNs', labelpad=labelpad)
        ax.set_xlabel('From ORNs', labelpad=labelpad)
        # ax.set_title(string)
        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=[vlim[0], vlim[1]])
        cb.outline.set_linewidth(0.5)
        cb.set_label('Weight', fontsize=7, labelpad=-7)
        plt.tick_params(axis='both', which='major', labelsize=7)
        plt.axis('tight')

        tools.save_fig(tools.get_experiment_name(modeldir),
                       tools.get_model_name(modeldir) + '_cutoff_' + string)

    w_orn = tools.load_pickle(modeldir)['w_orn']
    w_orn = tools.reshape_worn(w_orn, unique_orn=50)
    w_orn = w_orn.mean(axis=0)

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

    arg = tools.get_model_name(modeldir) + '_hist_'
    arg = arg + 'shuffled' if shuffle else arg
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.65, 0.6])
    plt.hist(all_gs, bins=50, range=[0, 1], align='left')
    plt.plot([cutoff, cutoff], [0, ax.get_ylim()[1]], '--', color='gray')
    ax.set_xlabel('GloScore')
    ax.set_ylabel('Count')
    ax.set_title('GloScore distribution for all PNs', fontsize=7)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.xlim([-0.05, 1.05])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    tools.save_fig(tools.get_experiment_name(modeldir), arg)
    return ix_good, ix_bad


def multiglo_pn2kc_distribution(modeldir, ix_good, ix_bad):
    # weights
    w_kc = tools.load_pickle(modeldir)['w_glo']
    sum_kc_weights = np.sum(w_kc, axis=1)
    weight_to_good = sum_kc_weights[ix_good]
    weight_to_bad = sum_kc_weights[ix_bad]

    arg = tools.get_model_name(modeldir) + '_pn2kc_hist_'
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
    tools.save_fig(tools.get_experiment_name(modeldir), arg)


def _lesion_multiglomerular_pn(path, units):
    import tensorflow as tf
    from model import FullModel

    config = tools.load_config(path)
    tf.reset_default_graph()

    # Load dataset
    config.dataset = 'proto'
    config.data_dir = './datasets/proto/standard'
    config.save_path = path
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

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


def multiglo_lesion(path, ix, ix_good, ix_bad):
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


def correlation_matrix(modeldir, arg='ortho', vlim=None):
    w_orn = tools.load_pickle(modeldir)['w_orn']

    out = np.zeros((w_orn.shape[1], w_orn.shape[1]))
    for i in range(w_orn.shape[1]):
        for j in range(w_orn.shape[1]):
            x = np.dot(w_orn[:, i], w_orn[:, j])
            y = np.corrcoef(w_orn[:, i], w_orn[:, j])[0, 1]
            if arg == 'ortho':
                out[i, j] = x
            elif arg == 'corr':
                out[i, j] = y
            else:
                raise ValueError('Unknown arg', arg)

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)

    max = np.max(abs(out))
    if not vlim:
        vlim = np.round(max, decimals=1) if max > .1 else np.round(max,
                                                                   decimals=2)
    im = ax.imshow(out, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none', origin='upper')

    title_txt = 'orthogonality' if arg == 'ortho' else 'correlation'
    plt.title('ORN-PN ' + title_txt)
    ax.set_xlabel('PN', labelpad=-5)
    ax.set_ylabel('PN', labelpad=-5)

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticks([0, out.shape[1]])
    ax.set_yticks([0, out.shape[0]])
    # plt.xlim([-.5, out.shape[1]+0.5])
    # plt.ylim([-.5, out.shape[1]+0.5])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')

    txt = '_' + title_txt + '_'
    tools.save_fig(tools.get_experiment_name(modeldir),
                   tools.get_model_name(modeldir) + txt)


def plot_distance_distribution(modeldir):
    """Plot distribution of cosine distance"""
    # Plot distribution of distance
    w_orn = tools.load_pickle(modeldir)['w_orn']
    # Average across neurons of same type
    w_orn = tools.reshape_worn(w_orn, w_orn.shape[1]).mean(axis=0)

    positive_w = w_orn.min() > 1e-6
    if positive_w:
        w_random = np.random.rand(w_orn.shape[0], 500)
    else:
        w_random = np.random.randn(w_orn.shape[0], 500)

    def _get_corr(data):
        """Get correlation bewteen data (N_sample, N_feature)."""
        similarity_matrix = cosine_similarity(data)  # (N_PN, N_PN)
        diag_mask = ~np.eye(similarity_matrix.shape[0], dtype=bool)
        corrs = similarity_matrix[diag_mask]
        return corrs

    corrs = _get_corr(w_orn.T)
    corrs_random = _get_corr(w_random.T)

    bins = np.linspace(-1, 1, 51)
    fig = plt.figure(figsize=(3, 1.5))
    ax = fig.add_axes([.2, .35, .6, .55])
    plt.hist(corrs, bins=bins, density=True, label='Trained', alpha=0.5)
    plt.hist(corrs_random, bins=bins, density=True, label='Random', alpha=0.5)
    ax.legend(fontsize=7, frameon=False)
    plt.xlabel('Cosine similarity')
    plt.ylabel('Distribution')
    plt.yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    figname = 'ORNPNCosineDistance'
    tools.save_fig(tools.get_experiment_name(modeldir),
                   tools.get_model_name(modeldir) + figname)
