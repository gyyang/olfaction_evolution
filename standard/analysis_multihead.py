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
sys.path.append(rootpath)

import task
from model import FullModel
import tools
import standard.analysis_pn2kc_training as analysis_pn2kc_training
from tools import save_fig

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

LABELS = ['Input degree', 'Conn. to valence', 'Conn. to identity']
RANGES = [(0, 15), (0, 5), (0, 10)]


def _get_data(path):
    """Load data.
    
    Returns:
        data: np array (n_neuron, dim)
            The rows are input degree, conn. to valence, conn. to identity
    """
    # TODO: clean up these paths
    # d = os.path.join(path, '000000', 'epoch')
    # d = os.path.join(path, 'epoch')
    d = path
    wout1 = tools.load_pickle(d, 'model/layer3/kernel:0')[-1]
    wout2 = tools.load_pickle(d, 'model/layer3_2/kernel:0')[-1]
    wglo = tools.load_pickle(d, 'w_glo')[-1]

    thres = analysis_pn2kc_training.infer_threshold(wglo)
    print('Inferred threshold', thres)
    sparsity = np.count_nonzero(wglo > thres, axis=0)
    strength_wout1 = np.linalg.norm(wout1, axis=1)
    strength_wout2 = np.linalg.norm(wout2, axis=1)
    data = np.stack([sparsity, strength_wout2, strength_wout1]).T

    return data


def _get_groups(data_norm, config, n_clusters=2):
    labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(data_norm)
    label_inds = np.arange(n_clusters)
    group_sizes = np.array([np.sum(labels==ind) for ind in label_inds])
    ind_sort = np.argsort(group_sizes)
    label_inds = [label_inds[i] for i in ind_sort]
    groups = [np.arange(config.N_KC)[labels==l] for l in label_inds]
    print('Group sizes', group_sizes[ind_sort])
    return groups


def _plot_scatter(v1, v2, xmin, xmax, ymin, ymax, xlabel, ylabel, figpath,
                  xticks = (0, 7, 15)):
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
    ax.scatter(v1, v2, alpha=0.2, marker='.', s=1)
    ax.set_xticks(xticks)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    save_fig(figpath, 'scatter_' + xlabel + '_' + ylabel)


def _compute_silouette_score(data, figpath, plot=True):
    n_clusters = np.arange(2, 10)
    scores = list()
    for n in n_clusters:
        labels = KMeans(n_clusters=n, random_state=0).fit_predict(data)
        score = silhouette_score(data, labels)
        scores.append(score)

    if plot:
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
        ax.plot(n_clusters, scores, 'o-')
        plt.xlabel('Number of clusters')
        plt.ylabel('Silouette score')
        plt.xticks([2, 5, 10])
        [ax.spines[s].set_visible(False) for s in ['right', 'top']]
        save_fig(figpath, 'silhouette_score')
    
    optim_n_clusters = n_clusters[np.argmax(scores)]
    return optim_n_clusters


def _get_density(data, X, Y, method='scipy'):
    """Get density of data.

    Args:
        data: array (n_samples, n_features)
    """
    positions = np.stack([X.ravel(), Y.ravel()]).T
    if method == 'scipy':
        # This method is most appropriate for unimodal distribution
        kernel = stats.gaussian_kde(data.T)
        Z = np.reshape(kernel(positions.T), X.shape)
    elif method == 'sklearn':
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(data)
        Z = np.reshape(np.exp(kde.score_samples(positions)), X.shape)
    else:
        raise ValueError('Unknown method')
    return Z


def _plot_density(Z, xind, yind, savename=None, figpath=None, title=None):
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.7])
    cmap = plt.cm.gist_earth_r
    # cmap = plt.cm.hot_r
    ax.imshow(np.rot90(Z), cmap=cmap,
              extent=list(RANGES[xind])+list(RANGES[yind]), aspect='auto')
    ax.set_xlim(RANGES[xind])
    ax.set_ylim(RANGES[yind])
    if xind == 0:
        ax.plot([7, 7], RANGES[yind], '--', color='gray', linewidth=1)
        ax.set_xticks([0, 7, 15])
    plt.xlabel(LABELS[xind])
    plt.ylabel(LABELS[yind])
    [ax.spines[s].set_visible(False) for s in ['left', 'right', 'top', 'bottom']]
    ax.tick_params(length=0)
    if title is not None:
        plt.title(title, fontsize=7)
    if savename is not None and figpath is not None:
        save_fig(figpath, savename)
    return fig, ax


def _plot_all_density(data, groups, xind, yind, figpath, normalize=True):
    data_plot = data[:, [xind, yind]]
    xmin, xmax = RANGES[xind]
    ymin, ymax = RANGES[yind]
    X_orig, Y_orig = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    if normalize:
        norm_mean = np.mean(data_plot, axis=0)
        norm_std = np.std(data_plot, axis=0)
        data_plot = (data_plot - norm_mean) / norm_std
        xmin, xmax = (np.array(RANGES[xind]) - norm_mean[0]) / norm_std[0]
        ymin, ymax = (np.array(RANGES[yind]) - norm_mean[1]) / norm_std[1]
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    Z = _get_density(data_plot, X, Y)
    Zs = [_get_density(data_plot[group], X, Y) for group in groups]
    
    def add_text(ax):
        for i in range(len(groups)):
            ind = np.argmax(Zs[i])
            ax.text(X_orig.flatten()[ind], Y_orig.flatten()[ind], str(i+1),
                    color='white')
        return ax

    name_pre = 'density_'+str(xind)+str(yind)
    fig, ax = _plot_density(Z, xind, yind)
    ax = add_text(ax)
    save_fig(figpath, name_pre)
    
    Zsum = 0
    for i, Z in enumerate(Zs):
        title = 'Cluster {:d} n={:d}'.format(i+1, len(groups[i]))
        _plot_density(Z, xind, yind, title=title)
        save_fig(figpath, name_pre+'_group'+str(i+1))
        Zsum += Z/Z.max()

    fig, ax = _plot_density(Zsum, xind, yind)
    ax = add_text(ax)
    save_fig(figpath, name_pre+'_group_sum')
    

def lesion_analysis(config, units=None):
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


def meta_lesion_analysis(config, units=None):
    import mamldataset
    import mamlmodel
    tf.reset_default_graph()
    num_samples_per_class = config.meta_num_samples_per_class
    num_class = config.N_CLASS
    dim_output = config.meta_output_dimension
    config.meta_batch_size = 100
    data_generator = mamldataset.DataGenerator(
        dataset= config.data_dir,
        batch_size=num_samples_per_class * num_class * 2,
        meta_batch_size=config.meta_batch_size,
        num_samples_per_class=num_samples_per_class,
        num_class=num_class,
        dim_output=dim_output)

    train_x_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                             data_generator.batch_size,
                                             config.N_PN))

    train_y_ph = tf.placeholder(tf.float32, (config.meta_batch_size,
                                             data_generator.batch_size,
                                             dim_output + config.n_class_valence))

    val_model = mamlmodel.MAML(train_x_ph, train_y_ph, config, training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_model.load()

        if units is not None:
            # val_model.lesion_units('model/layer3/kernel:0', units)
            # val_model.lesion_units('model/layer3_2/kernel:0', units)
            val_model.lesion_units('model/layer2/kernel:0', units, arg='inbound')

        val_x, val_y = data_generator.generate('val')
        val_acc, val_acc2 = sess.run(
                val_model.total_acc3, {train_x_ph: val_x, train_y_ph: val_y})
        return val_acc, val_acc2


def _plot_hist(name, ylim_heads, acc_plot, figpath):
    if name == 'head1':
        ylim = [ylim_heads[0], 1]
        title = 'Identity'
        savename = 'lesion_acc_head1'
    else:
        ylim = [ylim_heads[1], 1]  # replace with n_proto_valence
        title = 'Valence'
        savename = 'lesion_acc_head2'

    fs = 6
    width = 0.5
    fig = plt.figure(figsize=(1.2, 1.2))
    ax = fig.add_axes([0.35, 0.35, 0.6, 0.4])
    xlocs = np.arange(len(acc_plot))
    b0 = ax.bar(xlocs, acc_plot,
                width=width, edgecolor='none')
    ax.set_xticks(xlocs)
    group_names = [str(i+1) for i in range(len(acc_plot))]
    ax.set_xticklabels(['None'] + group_names)
    ax.set_xlabel('Lesioning cluster', fontsize=fs)
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
    save_fig(figpath, savename)



if __name__ == '__main__':
    # main1('multi_head', foldername='multi_head', subdir='000000')
    arg = 'multi_head'
    foldername='multi_head'
    # foldername='tmp_multi_head'
    subdir='000024'
    # subdir='000001'
# def main1(arg, foldername=None, subdir=None):
    
    if arg == 'metatrain':
        if foldername is None:
            foldername = 'metatrain'
        if subdir is None:
            subdir = '0'
        ylim_heads = (.5, .5)
    else:
        if foldername is None:
            foldername = 'multi_head'
        if subdir is None:
            subdir = '000000'
        ylim_heads = (0, .8)

    path = os.path.join(rootpath, 'files', foldername)
    figpath = os.path.join(rootpath, 'figures', foldername)
    
    res = tools.load_all_results(path)

    subpath = os.path.join(path, subdir)
    # subpath = path
    config = tools.load_config(subpath)
    print('Learning rate', config.lr)
    print('PN norm', config.pn_norm_pre)
    try:
        # dirty hack
        config.data_dir = rootpath + config.data_dir.split('olfaction_evolution')[1]
        config.save_path = rootpath + config.save_path.split('olfaction_evolution')[1]
    except IndexError:
        config.data_dir = rootpath + config.data_dir[1:]
        config.save_path = rootpath + config.save_path[1:]

    data = _get_data(subpath)
    
    norm_factor = data.mean(axis=0)
    # data_norm = data / norm_factor
    data_norm = (data - data.mean(axis=0)) / np.std(data, axis=0)
    
    optim_n_clusters = _compute_silouette_score(data_norm, figpath)
    # optim_n_clusters = 2
    groups = _get_groups(data_norm, config, n_clusters=optim_n_clusters)
    
# =============================================================================
#     _plot_scatter(v1, v2, xmin, xmax, ymin, ymax, xlabel=degree_label,
#                   ylabel=valence_label, figpath=figpath)
#     _plot_scatter(v1, v3, xmin, xmax, ymin, ymax, xlabel=degree_label,
#                   ylabel=class_label, figpath=figpath)
#     _plot_scatter(v3, v2, 0, 5, 0, 5, xlabel= class_label,
#                   ylabel=valence_label, figpath=figpath, xticks=[0, 1, 2, 3, 4, 5])
# =============================================================================
    
    _plot_all_density(data, groups, xind=0, yind=1, figpath=figpath)
    _plot_all_density(data, groups, xind=2, yind=1, figpath=figpath)
    
    val_accs = list()
    val_acc2s = list()
    for units in [None] + groups:
        if arg == 'metatrain':
            acc, acc2 = meta_lesion_analysis(config, units)
        elif arg == 'multi_head':
            acc, acc2 = lesion_analysis(config, units)
        val_accs.append(acc)
        val_acc2s.append(acc2)

    _plot_hist('head1', ylim_heads, val_accs, figpath)
    _plot_hist('head2', ylim_heads, val_acc2s, figpath)

