import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors.kde import KernelDensity
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import task
from model import FullModel
import tools
import standard.analysis_pn2kc_training as analysis_pn2kc_training
from standard.analysis import _easy_save

mpl.rcParams['font.size'] = 7

def main():
    foldername = 'multi_head'
    # foldername = 'tmp_train'

    path = os.path.join(rootpath, 'files', foldername)
    figpath = os.path.join(rootpath, 'figures', foldername)

    analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)

    # TODO: clean up these paths
    path = os.path.join(path, '000000')
    config = tools.load_config(path)
    config.data_dir = rootpath + config.data_dir[1:]
    config.save_path = rootpath + config.save_path[1:]

    # d = os.path.join(path, '000000', 'epoch')
    d = os.path.join(path, 'epoch')
    # Load results from last epoch
    wout1 = tools.load_pickle(d, 'model/layer3/kernel:0')[-1]
    wout2 = tools.load_pickle(d, 'model/layer3_2/kernel:0')[-1]
    wglo = tools.load_pickle(d, 'w_glo')[-1]

    # Compute sparsity
    thres = analysis_pn2kc_training.infer_threshold(wglo)
    sparsity = np.count_nonzero(wglo > thres, axis=0)


    # =============================================================================
    # ind_sort = np.argsort(sparsity)
    # sparsity = sparsity[ind_sort]
    # wout1 = wout1[ind_sort, :]
    # wout2 = wout2[ind_sort, :]
    # plt.figure()
    # plt.imshow(wout1[:500], aspect='auto')
    # plt.figure()
    # plt.imshow(wout2[:500], aspect='auto')
    # =============================================================================

    v1 = sparsity
    # strength_wout1 = np.sum(abs(wout1), axis=1)
    # strength_wout2 = np.sum(abs(wout2), axis=1)
    strength_wout2 = np.linalg.norm(wout2, axis=1)

    # v2 = strength_wout1
    v2 = strength_wout2
    # v2 = strength_wout2/(strength_wout1+strength_wout2)

    xlabel = 'PN Input degree'
    ylabel = 'Conn. to valence'
    xmin, xmax, ymin, ymax = 0, 15, 0, 3

    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
    ax.scatter(v1, v2, alpha=0.3, marker='.')
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    ax.set_xticks([0, 7, 15])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    _easy_save(figpath, 'scatter')

    data = np.stack([v1, v2]).T
    norm_factor = data.mean(axis=0)
    data_norm = data / norm_factor


    def _compute_silouette_score(data):
        n_clusters = np.arange(2, 10)
        scores = list()
        for n in n_clusters:
            labels = KMeans(n_clusters=n, random_state=0).fit_predict(data)
            score = silhouette_score(data, labels)
            scores.append(score)

        fig = plt.figure(figsize=(1.5, 1.5))
        plt.plot(n_clusters, scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silouette score')

    _compute_silouette_score(data_norm)

    labels = KMeans(n_clusters=2, random_state=0).fit_predict(data_norm)

    group0 = np.arange(config.N_KC)[labels==0]
    group1 = np.arange(config.N_KC)[labels==1]
    
    print('Group 0 has {:d} neurons'.format(len(group0)))
    print('Group 1 has {:d} neurons'.format(len(group1)))

    fig = plt.figure(figsize=(1.5, 1.5))
    plt.scatter(v1[group0], v2[group0], alpha=0.02)
    plt.scatter(v1[group1], v2[group1], alpha=0.3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.stack([X.ravel(), Y.ravel()]).T

    # positions /= norm_factor
    # data /= norm_factor

    def _get_density(data, method='scipy'):
        """Get density of data.

        Args:
            data: array (n_samples, n_features)
        """
        if method == 'scipy':
            kernel = stats.gaussian_kde(data.T)
            Z = np.reshape(kernel(positions.T), X.shape)
        elif method == 'sklearn':
            kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
            Z = np.reshape(np.exp(kde.score_samples(positions)), X.shape)
        else:
            raise ValueError('Unknown method')
        return Z

    Z = _get_density(data)
    Z1 = _get_density(data[group0])
    Z2 = _get_density(data[group1])

    def _plot_density(Z, savename):
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
        ax.plot([7, 7], [ymin, ymax], '--', color='gray', linewidth=1)
        ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r,
                  extent=[xmin, xmax, ymin, ymax], aspect='auto')
        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymin, ymax])
        ax.set_xticks([0, 7, 15])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        _easy_save(figpath, savename)


    _plot_density(Z, 'density')
    _plot_density(Z1, 'density_group1')
    _plot_density(Z2, 'density_group2')
    _plot_density(Z1+Z2, 'density_group12')


    def lesion_analysis(units=None):
        tf.reset_default_graph()

        # Load dataset
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
                val_model.lesion_units('model/layer3/kernel:0', units)
                val_model.lesion_units('model/layer3_2/kernel:0', units)

            # Validation
            val_loss, val_acc, val_acc2 = sess.run(
                [val_model.loss, val_model.acc, val_model.acc2],
                {val_x_ph: val_x, val_y_ph: val_y})

            return val_acc, val_acc2

    val_accs = list()
    val_acc2s = list()
    for units in [None, group0, group1]:
        acc, acc2 = lesion_analysis(units)
        val_accs.append(acc)
        val_acc2s.append(acc2)


    def _plot_hist(name):
        if name == 'head1':
            acc_plot = val_accs
            ylim = [0, 1]
            title = 'Odor'
            savename = 'lesion_acc_head1'
        else:
            acc_plot = val_acc2s
            ylim = [0.9, 1]  # replace with n_proto_valence
            title = 'Valence'
            savename = 'lesion_acc_head2'

        fs = 6
        width = 0.7
        fig = plt.figure(figsize=(1.2, 1.2))
        ax = fig.add_axes([0.35, 0.35, 0.6, 0.4])
        xlocs = np.arange(len(val_accs))
        b0 = ax.bar(xlocs, acc_plot,
                    width=width, edgecolor='none')
        ax.set_xticks(xlocs)
        ax.set_xticklabels(['None', 'Group 1', 'Group 2'], rotation=25)
        ax.set_xlabel('Lesioning', fontsize=fs, labelpad=-2)
        ax.set_ylabel('Accuracy', fontsize=fs)
        ax.set_title(title, fontsize=fs)
        ax.tick_params(axis='both', which='major', labelsize=fs)
        plt.locator_params(axis='y',nbins=2)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # ax.set_xlim([-0.8, len(rules_perf)-0.2])
        ax.set_ylim(ylim)
        ax.set_yticks(ylim)
        _easy_save(figpath, savename)


    _plot_hist('head1')
    _plot_hist('head2')

if __name__ == '__main__':
    main()