import os
import sys
import pickle
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from sklearn.linear_model import LinearRegression

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import save_fig
import dict_methods
from standard.analysis_weight import infer_threshold


figpath = os.path.join(rootpath, 'figures')
THRES = 0.1
FIGSIZE = (1.6, 1.2)
RECT = [0.3, 0.3, 0.65, 0.65]


def _set_colormap(nbins):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm


def compute_sparsity(d, epoch, dynamic_thres=False, visualize=False,
                     thres=THRES):
    print('compute sparsity needs to be replaced')
    wglos = tools.load_pickles(os.path.join(d, 'epoch'), 'w_glo')
    w = wglos[epoch]
    sparsity, thres = _compute_sparsity(w, dynamic_thres, visualize, thres)
    return sparsity


def _compute_sparsity(w, dynamic_thres=False, visualize=False, thres=THRES):
    w[np.isnan(w)] = 0

    # dynamically infer threshold after training
    if dynamic_thres is False:
        thres = thres
    elif dynamic_thres == True:
        thres = None
    else:
        thres = dynamic_thres
    thres, _ = infer_threshold(w, visualize=visualize, force_thres=thres)
    if dynamic_thres:
        print('dynamic thres = {:0.5f}'.format(thres))
    else:
        print('fixed thres = {:0.5f}'.format(thres))

    sparsity = np.count_nonzero(w > thres, axis=0)
    return sparsity, thres


def plot_sparsity(modeldir, epoch=None, xrange=50, plot=True):
    model_name = tools.get_model_name(modeldir)
    config = tools.load_config(modeldir)
    if epoch is not None and epoch != -1:
        modeldir = tools.get_modeldirs(os.path.join(modeldir, 'epoch'))[epoch]

    if (('kc_prune_weak_weights' in dir(config) and
         config.kc_prune_weak_weights)
            or ('prune_weak_weights' in dir(config) and
                config.prune_weak_weights)):
        prune = True
    else:
        prune = False

    log = tools.load_log(modeldir)
    if 'sparsity_inferred' not in log:
        w_glo = tools.load_pickle(modeldir)['w_glo']
        if prune:
            sparsity, thres_inferred = _compute_sparsity(
                w_glo, dynamic_thres=False, thres=config.kc_prune_threshold)
        else:
            sparsity, thres_inferred = _compute_sparsity(
                w_glo, dynamic_thres=True)
    else:
        sparsity = log['sparsity_inferred'][-1]

    if plot:
        if epoch is not None:
            string = '_epoch_' + str(epoch)
        else:
            string = ''

        save_path = os.path.join(figpath, tools.get_experiment_name(modeldir))
        save_name = os.path.join(save_path, '_' + model_name + '_sparsity' + string)
        _plot_sparsity(sparsity, save_name, xrange=xrange,
                       prune=prune)
    return sparsity


def _plot_sparsity(data, savename, xrange=50, yrange=None, prune=True):
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    plt.hist(data, bins=xrange, range=[0, xrange], density=True, align='left')
    if prune:
        xlabel = 'PN inputs per KC'
    else:
        xlabel = 'Strong PN inputs per KC'
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Fraction of KCs')

    hist, _ = np.histogram(data, bins=xrange, range=[0, xrange],
                           density=True)
    vmax = np.max(hist)
    if yrange is None:
        if vmax > 0.5:
            yrange = 1
        elif vmax > 0.25:
            yrange = 0.5
        else:
            yrange = 0.25

    xticks = [0, 5, 15, 25, 50]
    ax.set_xticks(xticks)
    ax.set_yticks([0, yrange])
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange])
    # Add text
    plt.text(np.mean(data), vmax * 1.1, r'K = {:0.1f} ({:0.1f})'.format(
        np.mean(data), np.std(data)))

    # plt.title(data[data>0].mean())

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    split = os.path.split(savename)
    tools.save_fig(split[0], split[1])


def plot_sparsity_movie(modeldir):
    log = tools.load_log(modeldir)

    xrange = 50
    yrange = 0.5

    data = log['sparsity_inferred'][0]
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)

    hist, bins = np.histogram(data, bins=xrange, range=[0, xrange], density=True)
    rects = plt.bar((bins[:-1]+bins[1:])/2, hist)
    plt.plot([7, 7], [0, yrange], '--', color='gray')
    ax.set_xlabel('PN inputs per KC')
    ax.set_ylabel('Fraction of KCs')

    xticks = [1, 7, 15, 25, 50]
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(0, yrange, 3))
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange])
    # plt.title(data[data > 0].mean())

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    title = ax.text(0.35, 1.05, "", transform=ax.transAxes, ha='left')
    # animation function.  This is called sequentially
    def animate(i):
        hist, bins = np.histogram(log['sparsity_inferred'][i],
                                  bins=xrange, range=[0, xrange],
                                  density=True)
        for rect, h in zip(rects, hist):
            rect.set_height(h)
        title.set_text('Epoch ' + str(i).rjust(3))
        return rects

    # call the animator.  blit=True means only re-draw the parts that have changed.
    n_time = log['sparsity_inferred'].shape[0]
    anim = animation.FuncAnimation(fig, animate,
                                   frames=n_time, interval=20, blit=True)
    writer = animation.writers['ffmpeg'](fps=30)
    split = os.path.split(modeldir)
    figname = tools.get_figname(split[0], split[1])
    anim.save(figname + 'sparsity_movie.mp4', writer=writer, dpi=200)


def plot_distribution(modeldir, epoch=None, xrange=1.0, **kwargs):
    """Plot weight distribution from a single model path."""
    model_name = tools.get_model_name(modeldir)
    config = tools.load_config(modeldir)
    if epoch is not None:
        modeldir = tools.get_modeldirs(os.path.join(modeldir, 'epoch'))[epoch]

    w = tools.load_pickles(modeldir, 'w_glo')[0]
    w[np.isnan(w)] = 0
    distribution = w.flatten()

    if epoch is not None:
        string = '_epoch_' + str(epoch)
    else:
        string = ''

    if epoch == 0:
        thres, res_fit = None, None
    else:
        thres, res_fit = infer_threshold(distribution)
        if 'kc_prune_weak_weights' in dir(config) \
                and config.kc_prune_weak_weights:
            thres = config.kc_prune_threshold

    save_path = os.path.join(figpath, tools.get_experiment_name(modeldir))
    save_name = os.path.join(save_path, '_' + model_name + '_')
    _plot_distribution(
        distribution, save_name + 'distribution' + string,
        thres=thres, xrange=xrange, **kwargs)
    _plot_log_distribution(
        distribution, save_name + 'log_distribution' + string,
        thres=thres, res_fit=res_fit, **kwargs)


def _plot_log_distribution(data, savename, thres=0, res_fit=None, **kwargs):
    x = np.log(data + 1e-10)

    xticks = ['$10^{-6}$','$10^{-4}$', '.01', '1']
    xticks_log = np.log([1e-6, 1e-4, 1e-2, 1])

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    if res_fit is not None:
        plt.hist(x, bins=50, range=[-12, 3], density=True)
    else:
        weights = np.ones_like(x) / float(len(x))
        plt.hist(x, bins=50, range=[-12, 3], weights=weights)

    ax.set_xlabel('PN to KC Weight')
    ax.set_ylabel('Dist. of Connections')
    ax.set_xticks(xticks_log)
    ax.set_xticklabels(xticks)

    # yticks = [0, 1000, 2000, 3000, 4000, 5000]
    # yticklabels = ['0', '1K', '2K', '3K', '4K', '>100K']
    # ax.set_yticks(yticks)
    # ax.set_yticklabels(yticklabels)
    # plt.ylim([0, yrange])
    plt.xlim([-12, 3])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    if thres is not None:
        thres = np.log(thres)
        ax.plot([thres, thres], [0, plt.ylim()[1]], '--', color='gray', linewidth=1)

    if res_fit is not None:
        for i in range(res_fit['n_modal']):
            ax.plot(res_fit['x_plot'], res_fit['pdfs'][i],
                    linestyle='--', linewidth=1, alpha=1)

    split = os.path.split(savename)
    tools.save_fig(split[0], split[1])


def _plot_distribution(data, savename, xrange=None, yrange=None, broken_axis=True,
                       thres=None, approximate=True):
    fig = plt.figure(figsize=FIGSIZE)
    if not broken_axis:
        if yrange is None:
            yrange = 5000
        ax = fig.add_axes(RECT)
        plt.hist(data, bins=50, range=[0, xrange], density=False)
        ax.set_xlabel('PN to KC Weight')
        ax.set_ylabel('Number of Connections')

        xticks = [0, .2, .4, .6, .8, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        yticks = [0, 1000, 2000, 3000, 4000, 5000]
        yticklabels = ['0', '1K', '2K', '3K', '4K', '>100K']
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if thres is not None:
            ax.plot([thres, thres], [0, yrange], '--', color='gray')
    else:
        if xrange is None:
            xrange = np.max(data)
        ax = fig.add_axes([0.3, 0.3, 0.65, 0.45])
        ax2 = fig.add_axes([0.3, 0.8, 0.65, 0.1])
        n, bins, _ = ax2.hist(data, bins=50, range=[0, xrange], density=False)
        ax.hist(data, bins=50, range=[0, xrange], density=False)

        # hide the spines between ax and ax2
        ax2.spines['bottom'].set_visible(False)
        ax2.spines["top"].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax2.set_xticks([])
        ax2.xaxis.set_ticks_position('none')
        ax2.tick_params(labeltop='off')  # don't put tick labels at the top
        ax.xaxis.tick_bottom()

        d = .01  # how big to make the diagonal lines in axes coordinates
        # arguments to pass to plot, just so we don't keep repeating them
        kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
        ax2.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
        kwargs.update(transform=ax.transAxes)  # switch to the bottom axes
        ax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal

        ax.set_xlabel('PN to KC Weight')
        ax.set_ylabel('Number of Conn.')

        xticks = np.arange(0, xrange + 0.01, .5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])

        if thres is None:
            yrange = 5000
            yticks = [0, 2500, 5000]
            yticklabels = ['0', '2.5K', '5K']
        else:
            ymax = np.round(np.max(n[bins[:-1]>thres]) * 2/1000, decimals=1)
            yrange = 1000*ymax
            yticks = [0, yrange]
            yticklabels = ['0', '{:s}K'.format(str(ymax))]

        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(0, yrange)  # most of the data

        ax2.set_yticks([np.max(n)])
        ax2.set_yticklabels(['{:d}K'.format(int(np.max(n)/1000))])
        ax2.set_ylim(0.9 * np.max(n), 1.1 * np.max(n))  # outliers only
        if thres is not None:
            ax.plot([thres, thres], [0, yrange], '--', color='gray')
            ax2.plot([thres, thres], ax2.get_ylim(), '--', color='gray')

        split = os.path.split(savename)
        tools.save_fig(split[0], split[1])


def plot_log_distribution_movie(modeldir):
    log = tools.load_log(modeldir)

    xticks = ['$10^{-6}$','$10^{-4}$', '.01', '1']
    xticks_log = np.log([1e-6, 1e-4, 1e-2, 1])

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)

    xdata, ydata = log['log_bins'][:-1], log['log_hist'][0]

    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_axes(RECT)
    ln, = ax.plot(xdata, ydata)

    ax.set_xlabel('PN to KC Weight')
    ax.set_ylabel('Distribution of Connections')
    ax.set_xticks(xticks_log)
    ax.set_xticklabels(xticks)

    ymax = np.max(log['log_hist'][1:]) * 1.2
    plt.ylim([0, ymax])
    plt.xlim([-12, 3])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    title = ax.text(0.35, 1.05, "", transform=ax.transAxes, ha='left')

    # animation function.  This is called sequentially
    def animate(i):
        ln.set_data(xdata, log['log_hist'][i])
        title.set_text('Epoch ' + str(i).rjust(3))
        return ln,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    n_time = log['log_hist'].shape[0]
    anim = animation.FuncAnimation(fig, animate,
                                   frames=n_time, interval=20, blit=True)
    writer = animation.writers['ffmpeg'](fps=30)
    split = os.path.split(modeldir)
    figname = tools.get_figname(split[0], split[1])
    anim.save(figname + 'log_distribution_movie.mp4', writer=writer, dpi=600)


def _get_K_vs_N(name_or_path):
    """Get K vs N data for a name or path.

    Returns:
        n_orns: array like
        Ks: array like
    """
    if name_or_path == 'dim':
        # Get optimal K for high dimensionality as in Litwin-Kumar 17
        from analytical.analyze_simulation_results import _load_result
        n_orns, Ks = _load_result('all_value_withdim_m', v_name='dim')
    elif name_or_path == 'angle':
        # Optimal K for smallest angle change
        fname = os.path.join(rootpath, 'files', 'analytical',
                             'control_coding_level_summary')
        summary = pickle.load(open(fname, "rb"))
        # summary: 'opt_ks', 'coding_levels', 'conf_ints', 'n_orns'
        n_orns = summary['n_orns']
        Ks = summary['opt_ks'].T
    else:
        # Should be a path
        path = name_or_path

        acc_min = 0.
        path = path + '_pn'  # folders named XX_pn50, XX_pn100, ..
        folders = glob.glob(path + '*')
        n_orns = sorted([int(folder.split(path)[-1]) for folder in folders])
        Ks = list()
        new_n_orns = list()
        for n_orn in n_orns:
            _path = path + str(n_orn)
            modeldirs = tools.get_modeldirs(_path, acc_min=acc_min)
            modeldirs = tools.filter_modeldirs(
                modeldirs, exclude_badkc=True, exclude_badpeak=True)
            if len(modeldirs) == 0:
                continue
            # TODO: for meta need to be meta_lr
            # TODO: Temporary disable sorting
            # Use model with highest LR among good models
            # modeldirs = tools.sort_modeldirs(modeldirs, 'lr')
            # modeldirs = [modeldirs[-1]]

            res = tools.load_all_results(modeldirs)
            Ks.append(res['K_smart'])
            new_n_orns.append(n_orn)
        n_orns = np.array(new_n_orns)
    return n_orns, Ks


def _plot_K_vs_N(ax, name_or_path, results=None, plot_box=True, plot_fit=True):
    """Plot one set of data."""
    if name_or_path == 'data':
        ax.plot(np.log(1000), np.log(100), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(1000), np.log(120), '[2]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='bottom',
                zorder=5)

        ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(1000), np.log(32), '[3]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='top',
                zorder=5)
        ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(53), np.log(6), '[1]', color=tools.darkblue,
                horizontalalignment='left', verticalalignment='top', zorder=5)
        return ax

    # Get data
    if results is None:
        results = dict()
    if name_or_path in results.keys():
        n_orns, Ks = results[name_or_path]
    else:
        n_orns, Ks = _get_K_vs_N(name_or_path)

    def _fit(x, y):
        x_fit = np.linspace(min(np.log(20), x[0]), max(np.log(1200), x[-1]), 3)
        # model = Ridge()
        model = LinearRegression()
        model.fit(x[:, np.newaxis], y)
        y_fit = model.predict(x_fit[:, np.newaxis])
        return x_fit, y_fit, model

    logKs = [np.log(K) for K in Ks]
    med_logKs = np.array([np.median(np.log(K)) for K in Ks])

    name_or_path = Path(name_or_path).name
    if name_or_path == 'dim':
        color = tools.gray
    elif name_or_path == 'vary_or':
        color = tools.blue
    elif name_or_path == 'meta_vary_or':
        color = tools.red
    else:
        color = tools.gray

    if name_or_path == 'dim':
        ax.scatter(np.log(n_orns), logKs, s=2., c=color)
    else:
        if plot_box:
            tools.pretty_box(logKs, np.log(n_orns), ax, color)

    if plot_fit:
        x_fit, y_fit, model = _fit(np.log(n_orns), med_logKs)
        label = tools.nicename(name_or_path, mode='scaling')
        label = label + r' $K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])
        ax.plot(x_fit, y_fit, color=color, label=label)
    return ax


def plot_all_K(name_or_paths, *args):
    """Plot the typical K-N plot.

    Args:
       name_or_paths: a list of str
    """
    if isinstance(name_or_paths, str):
        name_or_paths = [name_or_paths]

    all_name_or_paths = name_or_paths[:]
    for arg in args:
        all_name_or_paths += arg
    all_name_or_paths = set(all_name_or_paths)

    # (n_orns, Ks)
    results = {n: _get_K_vs_N(n) for n in all_name_or_paths}

    def _plot_all_K(name_or_paths):
        fig = plt.figure(figsize=(4, 2.5))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        for name_or_path in name_or_paths:
            ax = _plot_K_vs_N(ax, name_or_path, results)

        ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)

        ax.set_xlabel('Number of ORs (N)')
        ax.set_ylabel('Expansion Input Degree (K)')
        xticks = np.array([25, 50, 100, 200, 500, 1000])
        ax.set_xticks(np.log(xticks))
        ax.set_xticklabels([str(t) for t in xticks])
        yticks = np.array([3, 10, 30, 100])
        ax.set_yticks(np.log(yticks))
        ax.set_yticklabels([str(t) for t in yticks])
        ax.set_xlim(np.log([15, 1700]))
        ax.set_ylim(np.log([2, 300]))
        ax.grid(True, alpha=0.5)

        name = '.'.join([Path(n).name for n in name_or_paths])
        # All save to the same directory
        save_fig('scaling', name)

    _plot_all_K(name_or_paths)
    for arg in args:
        _plot_all_K(arg)

