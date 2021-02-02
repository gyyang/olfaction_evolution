import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors.kde import KernelDensity

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import task
import tools
import standard.analysis_weight as analysis_weight
from tools import save_fig
import settings


LABELS = ['Input degree', 'Conn. to valence', 'Conn. to identity']
TICKS = [(1, 7, 15), None, None]
RANGES = [(0, 15), (0, 7), (0, 10)]
LOGTICKS = [(1, 7, 50), (0.01, 0.1, 1, 10), (0.1, 1, 10)]
LOGRANGES = [(1, 50), (0.01, 10), (0.1, 10)]


def _fix_config(config):
    """Hack function to fix config."""
    try:
        # dirty hack
        config.data_dir = rootpath + config.data_dir.split('olfaction_evolution')[1]
        config.save_path = rootpath + config.save_path.split('olfaction_evolution')[1]
    except IndexError:
        config.data_dir = rootpath + config.data_dir[1:]
        config.save_path = rootpath + config.save_path[1:]
    return config


def _get_data(modeldir, use_log=True, exclude_weak=True):
    """Prepare data for multihead analysis.
    
    Returns:
        data: np array (n_neuron, dim)
            The columns are input degree, conn. to valence, conn. to identity
        data_norm: z-score normalized data array
        ignore_indices: np boolean array shape (n_neuron,) whether to ignore
    """
    if settings.use_torch:
        res = tools.load_pickle(modeldir)
        wout1, wout2 = res['w_out'], res['w_out2']
    else:
        wout1 = tools.load_pickles(modeldir, 'model/layer3/kernel:0')[-1]
        wout2 = tools.load_pickles(modeldir, 'model/layer3_2/kernel:0')[-1]

    wglo = tools.load_pickles(modeldir, 'w_glo')[-1]
    config = tools.load_config(modeldir)

    if config.kc_prune_weak_weights:
        thres = config.kc_prune_threshold
        print('Using KC prune threshold')
    else:
        thres, _ = analysis_weight.infer_threshold(wglo)
        print('Inferred threshold', thres)
    sparsity = np.count_nonzero(wglo > thres, axis=0)
    strength_wout1 = np.linalg.norm(wout1, axis=1)
    strength_wout2 = np.linalg.norm(wout2, axis=1)
    # This ordering is important
    data = np.stack([sparsity, strength_wout2, strength_wout1]).T

    ignore_indices = sparsity == 0  # ignore neurons with no inputs

    if use_log:
        # Use log data
        data[data == 0.] = np.exp(-5) # Prevent zeros in data_plot
        data = np.log(data)

    if exclude_weak:
        thres = -2  # This is empirical, depends on w strength definition
        if not use_log:
            thres = np.exp(thres)
        # TODO: Find a better way to determine threshold
        ignore_indices = np.logical_or(
            ignore_indices, np.mean(data[:, [1, 2]], axis=1) < thres)


    data_norm = (data - data.mean(axis=0)) / np.std(data, axis=0)

    results = {'data': data, 'data_norm': data_norm,
               'ignore_indices': ignore_indices}
    return results


def _compute_silouette_score(results, figpath=None, plot=False, name_pre=None):
    """Compute silouette score.

    Args:
        data: (n_neuron, dim), dim should be 3

    Returns:
        optim_n_clusters: int
    """
    data = results['data']
    data_norm = results['data_norm']
    ignore_indices = results['ignore_indices']

    data_norm_use = data_norm[~ignore_indices]

    n_clusters = np.arange(2, 10)
    scores = list()
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n, random_state=0)
        labels = kmeans.fit_predict(data_norm_use)
        score = silhouette_score(data_norm_use, labels)
        scores.append(score)

    if plot:
        fig = plt.figure(figsize=(1.5, 1.5))
        ax = fig.add_axes([0.3, 0.3, 0.65, 0.65])
        ax.plot(n_clusters, scores, 'o-', markersize=3)
        plt.xlabel('Number of clusters')
        plt.ylabel('Silouette score')
        plt.xticks([2, 5, 10])
        [ax.spines[s].set_visible(False) for s in ['right', 'top']]
        name = 'silhouette_score'
        if name_pre is not None:
            name = name_pre + '_' + name
        save_fig(figpath, name)
        plt.close()

    optim_n_clusters = n_clusters[np.argmax(scores)]
    return optim_n_clusters


def _get_groups(results, n_clusters=2, sortmode='head2'):
    """Get groups based on data_norm.

    Args:
        results: result dict from _get_data
            data_norm: np array (n_units, n_features)
        n_clusters: int, number of clusters to use
        sortmode: str, 'size' or 'head2', mode for sorting groups

    Returns:
        groups: list of np index arrays
    """
    data = results['data']
    data_norm = results['data_norm']
    ignore_indices = results['ignore_indices']

    use_indices = ~ignore_indices
    original_indices = np.arange(data.shape[0])[use_indices]

    data_norm_use = data_norm[use_indices]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data_norm_use)
    # outputs

    label_inds = np.arange(n_clusters)
    group_sizes = np.array([np.sum(labels == ind) for ind in label_inds])

    # Reorder the groups
    if sortmode == 'size':
        # Sort by size
        ind_sort = np.argsort(group_sizes)[::-1]
    elif sortmode == 'head2':
        # Sort by connection strength to head 2
        strength_wout2 = [np.mean(data_norm_use[labels == ind, 1]) for ind in
                          label_inds]
        ind_sort = np.argsort(strength_wout2)
    else:
        raise ValueError('Unknown sort mode', sortmode)
    group_sizes = group_sizes[ind_sort]
    label_inds = [label_inds[i] for i in ind_sort]

    # Get each group
    groups = [np.arange(len(labels))[labels == l] for l in label_inds]
    print('Group sizes', group_sizes)

    for group, group_size in zip(groups, group_sizes):
        assert len(group) == group_size

    # Obtain original indices
    groups = [original_indices[group] for group in groups]

    # if exclude_weak:
    #     new_groups = list()
    #     for group in groups:
    #         norm = np.mean(data[group, :][:, [1, 2]])  # norm of two heads
    #         if norm > -2:  # this threshold is a bit arbitrary
    #             new_groups.append(group)
    #         else:
    #             results['ignore_indices'][group] = True
    #     groups = new_groups

    if n_clusters != len(groups):
        print('Updating n clusters to ', len(groups))
        n_clusters = len(groups)  # update optim n clusters

    results['groups'] = groups
    results['n_clusters'] = n_clusters

    return results


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


def _plot_density(Z, xind, yind, savename=None, figpath=None, title=None, plot_log=False):
    fig = plt.figure(figsize=(1.6, 1.6))
    ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
    cmap = plt.cm.gist_earth_r
    # cmap = plt.cm.hot_r
    # ax.imshow(np.rot90(Z), cmap=cmap, aspect='auto')
    ranges = np.log(LOGRANGES) if plot_log else RANGES
    ax.imshow(np.rot90(Z), cmap=cmap,
              extent=list(ranges[xind])+list(ranges[yind]), aspect='auto')
    ax.set_xlim(ranges[xind])
    ax.set_ylim(ranges[yind])

    if xind == 0:
        vert_x = np.log(7) if plot_log else 7
        # ax.plot([vert_x, vert_x], ranges[yind], '--', color='gray', linewidth=1)

    if plot_log:
        ax.set_xticks(np.log(LOGTICKS[xind]))
        ax.set_xticklabels(LOGTICKS[xind])
    else:
        if TICKS[xind]:
            ax.set_xticks(TICKS[xind])

    if plot_log:
        ax.set_yticks(np.log(LOGTICKS[yind]))
        ax.set_yticklabels(LOGTICKS[yind])
    else:
        if TICKS[yind]:
            ax.set_yticks(TICKS[yind])

    plt.xlabel(LABELS[xind])
    plt.ylabel(LABELS[yind])
    [ax.spines[s].set_visible(False) for s in ['left', 'right', 'top', 'bottom']]
    ax.tick_params(length=0)
    if title is not None:
        plt.title(title, fontsize=7)
    if savename is not None and figpath is not None:
        save_fig(figpath, savename)
    return fig, ax


def _plot_all_density(results, xind, yind, figpath=None,
                      normalize=True, name_pre=None, plot_log=True,
                      plot_sum_only=False):
    """Plot density.

    Args:
        data: np array (n_points, n_features)
        groups: list of np arrays
        plot_log: bool, if True, data is in log form
    """
    data = results['data']
    data_norm = results['data_norm']
    ignore_indices = results['ignore_indices']
    groups = results['groups']

    data_plot = data[:, [xind, yind]]
    data_plot_kept = data_plot[~ignore_indices]
    if plot_log:
        xmin, xmax = np.log(LOGRANGES[xind])
        ymin, ymax = np.log(LOGRANGES[yind])
    else:
        xmin, xmax = RANGES[xind]
        ymin, ymax = RANGES[yind]

    X_orig, Y_orig = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    if normalize:
        norm_mean = np.mean(data_plot_kept, axis=0)
        norm_std = np.std(data_plot_kept, axis=0)
        print(norm_std)
        data_plot = (data_plot - norm_mean) / norm_std
        data_plot_kept = (data_plot_kept - norm_mean) / norm_std
        xmin, xmax = (np.array([xmin, xmax]) - norm_mean[0]) / norm_std[0]
        ymin, ymax = (np.array([ymin, ymax]) - norm_mean[1]) / norm_std[1]
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    
    def add_text(ax):
        for i in range(len(groups)):
            ind = np.argmax(Zs[i])
            ax.text(X_orig.flatten()[ind]+0.3, Y_orig.flatten()[ind], str(i+1),
                    color='darkblue')
        return ax

    # original density
    if name_pre is None:
        name_pre = ''
    name_pre = name_pre + 'density_'+str(xind)+str(yind)
    if not plot_sum_only:
        Z = _get_density(data_plot_kept, X, Y)
        fig, ax = _plot_density(Z, xind, yind, plot_log=plot_log)
        ax = add_text(ax)
        if figpath:
            save_fig(figpath, name_pre)

    # individual cluster normalized density
    # Here group indices are original ones, so use original data_plot
    Zs = list()
    for j, group in enumerate(groups):
        try:
            Zs.append(_get_density(data_plot[group], X, Y))
        except np.linalg.LinAlgError:
            # when neurons have the same input degree, density is degenerate
            Zs.append(_get_density(data_plot[group], X, Y, method='sklearn'))

    Zsum = 0
    for i, Z in enumerate(Zs):
        title = 'Cluster {:d} n={:d}'.format(i+1, len(groups[i]))
        if not plot_sum_only:
            _plot_density(Z, xind, yind, title=title, plot_log=plot_log)
            if figpath:
                save_fig(figpath, name_pre+'_group'+str(i+1))
        Zsum += Z/Z.max()

    # sum of multiple normalized density
    fig, ax = _plot_density(Zsum, xind, yind, plot_log=plot_log)
    ax = add_text(ax)
    if figpath:
        save_fig(figpath, name_pre+'_cluster'+str(
            results['n_clusters'])+'_group_sum')


def analyze_example_network(modeldir, arg='multi_head', fix_cluster=None):
    """Analyze example network for multi-head analysis."""
    model_name = tools.get_model_name(modeldir)

    figpath = os.path.join(rootpath, 'figures',
                           tools.get_experiment_name(modeldir))

    with open(os.path.join(modeldir, 'analysis_results.pkl'), 'rb') as f:
        results = pickle.load(f)

    name_pre = model_name
    _plot_all_density(results, xind=0, yind=1,
                      figpath=figpath, name_pre=name_pre, plot_sum_only=True)
    _plot_all_density(results, xind=2, yind=1,
                      figpath=figpath, name_pre=name_pre, plot_sum_only=True)

    # if arg == 'metatrain':
    #     ytick_heads = (.5, .5)
    # else:
    #     ytick_heads = (0, .8)
    # val_accs, val_acc2s = _get_lesion_acc(modeldir, groups, arg=arg)
    # for head, val_acc in zip(['head1', 'head2'], [val_accs, val_acc2s]):
    #     fig = _plot_hist(val_acc[:, np.newaxis], head, ytick_heads)
    #     save_fig(figpath,
    #              model_name+'_cluster'+str(optim_n_clusters)+'_lesion'+head)


def analyze_networks_lesion(modeldirs, arg='multi_head'):
    if arg == 'metatrain':
        ytick_heads = (.5, .5)
    else:
        ytick_heads = (0, .8)

    figpath = os.path.join(rootpath, 'figures',
                           tools.get_experiment_name(modeldirs[0]))

    val_accs, val_acc2s = list(), list()
    for modeldir in modeldirs:
        with open(os.path.join(modeldir, 'analysis_results.pkl'), 'rb') as f:
            results = pickle.load(f)
        groups = results['groups']
        if len(groups) != 2:
            # Merge all clusters before the last one
            # This is done so we can get two clusters for lesioning
            # Not a general method, but works here
            results['groups'] = [np.concatenate(groups[:-1]), groups[-1]]

        val_accs_tmp, val_acc2s_tmp = _get_lesion_acc(
            modeldir, results['groups'], arg=arg)

        val_accs.append(val_accs_tmp)
        val_acc2s.append(val_acc2s_tmp)
        
    val_accs = np.array(val_accs).T
    val_acc2s = np.array(val_acc2s).T

    for head, val_acc in zip(['head1', 'head2'], [val_accs, val_acc2s]):
        fig = _plot_hist(val_acc, head, ytick_heads)
        save_fig(figpath, 'population_lesion'+head)


def analyze_networks(modeldirs):
    for modeldir in modeldirs:
        print(modeldir)
        config = tools.load_config(modeldir)
        print(config.pn_norm_pre, config.kc_dropout_rate, config.lr)
        results = _get_data(modeldir)
        optim_n_clusters = _compute_silouette_score(results)
        results = _get_groups(results, n_clusters=optim_n_clusters)
        with open(os.path.join(modeldir, 'analysis_results.pkl'), 'wb') as f:
            pickle.dump(results, f)


def lesion_analysis(config, units=None):
    import settings
    if settings.use_torch:
        from standard.analysis_activity import load_activity_torch
        lesion_kwargs = {'layer3': units, 'layer3_2': units}
        results = load_activity_torch(config.save_path,
                                      lesion_kwargs=lesion_kwargs)
        return results['acc1'], results['acc2']
    else:
        # TODO: replace with load activity
        return lesion_analysis_tf(config, units)


def lesion_analysis_tf(config, units=None):
    import tensorflow as tf
    from model import FullModel

    tf.reset_default_graph()

    # Load dataset
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
            val_model.lesion_units('model/layer3/kernel:0', units)
            val_model.lesion_units('model/layer3_2/kernel:0', units)

        # Validation
        val_loss, val_acc, val_acc2 = sess.run(
            [val_model.loss, val_model.acc, val_model.acc2],
            {val_x_ph: val_x, val_y_ph: val_y})
        return (val_acc, val_acc2)


def meta_lesion_analysis(config, units=None):
    import tensorflow as tf
    import mamldataset
    import mamlmodel
    tf.reset_default_graph()
    num_samples_per_class = config.meta_num_samples_per_class
    num_class = config.N_CLASS
    dim_output = config.meta_output_dimension
    config.meta_batch_size = 100
    data_generator = mamldataset.DataGenerator(
        dataset=config.data_dir,
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
            val_model.lesion_units('model/layer2/kernel:0', units,
                                   arg='inbound')

        val_x, val_y = data_generator.generate('val')
        val_acc, val_acc2 = sess.run(
            val_model.total_acc3, {train_x_ph: val_x, train_y_ph: val_y})
        return (val_acc, val_acc2)


def _get_lesion_acc(modeldir, groups, arg='multi_head'):
    config = tools.load_config(modeldir)
    config = _fix_config(config)
    val_accs = list()
    val_acc2s = list()
    for units in [None] + groups:
        if arg == 'metatrain':
            acc, acc2 = meta_lesion_analysis(config, units)
        elif arg == 'multi_head':
            acc, acc2 = lesion_analysis(config, units)
        val_accs.append(acc)
        val_acc2s.append(acc2)

    val_accs = np.array(val_accs)
    val_acc2s = np.array(val_acc2s)
    return val_accs, val_acc2s


def _plot_hist(acc_plot, name, ytick_heads, plot_bar=False, plot_box=True):
    """
    Plot histogram.

    Args:
        name: head1 or head2
        ytick_heads: ytick for head 1 and head 2
        acc_plot: np array (n_box, n_point_per_box)
    """
    if name == 'head1':
        ytick = [ytick_heads[0], 1]
        title = 'Identity'
    else:
        ytick = [ytick_heads[1], 1]  # replace with n_proto_valence
        title = 'Valence'

    width = 0.5
    fig = plt.figure(figsize=(1.2, 1.2))
    ax = fig.add_axes([0.35, 0.35, 0.6, 0.4])
    xlocs = np.arange(len(acc_plot))
    if plot_bar:
        b0 = ax.bar(xlocs, acc_plot,
                    width=width, edgecolor='none', facecolor=tools.blue)
    if plot_box:
        color = tools.blue
        flierprops = {'markersize': 3, 'markerfacecolor': color,
                      'markeredgecolor': 'none'}
        boxprops = {'facecolor': color, 'linewidth': 1, 'color': color}
        medianprops = {'color': color * 0.5}
        whiskerprops = {'color': color}
        ax.boxplot(list(acc_plot), positions=xlocs, widths=width,
                   patch_artist=True, medianprops=medianprops,
                   flierprops=flierprops, boxprops=boxprops, showcaps=False,
                   whiskerprops=whiskerprops
                   )
    ax.set_xticks(xlocs)
    group_names = [str(i + 1) for i in range(len(acc_plot) - 1)]
    ax.set_xticklabels(['None'] + group_names)
    ax.set_xlabel('Lesioning cluster')
    ax.set_ylabel('Accuracy')
    ax.set_title(title, fontsize=7)
    ax.tick_params(axis='both', which='major')
    plt.locator_params(axis='y', nbins=2)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    # ax.set_xlim([-0.8, len(rules_perf)-0.2])
    dytick = (ytick[1] - ytick[0]) * 0.05
    ylim = ytick[0] - dytick, ytick[1] + dytick
    ax.set_ylim(ylim)
    ax.set_yticks(ytick)
    return fig