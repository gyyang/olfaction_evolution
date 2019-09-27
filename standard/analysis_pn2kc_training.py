import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from tools import nicename
from standard.analysis import _easy_save
import standard.analysis as sa
from scipy.stats import rankdata
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kurtosis
from scipy.stats import multivariate_normal

from sklearn.mixture import GaussianMixture


rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'
figpath = os.path.join(rootpath, 'figures')
THRES = 0.08

def _set_colormap(nbins):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm



def infer_threshold(x, use_logx=True, visualize=False, force_thres=None,
                    downsample=True):

    """Infers the threshold of a bi-modal distribution.

    The log-input will be fit as a mixture of 2 gaussians.

    Args:
        x: an array containing the values to be fitted
        use_logx: bool, if True, fit log-input

    Returns:
        thres: a scalar threshold that separates the two gaussians
    """
    # TODO: Make this function more robust
    # Select neurons that receive both strong and weak connections
    # weak connections should be around median, where strong should be around max
    x = np.array(x)
    ratio = np.max(x, axis=0) / np.median(x, axis=0)
    # heuristic that works well for N=50-500, can plot hist of ratio
    ind = ratio > 15
    x = x[:, ind]  # select expansion layer neurons
    x = x.flatten()

    if downsample:
        if len(x) > 1e5:
            x = np.random.choice(x, size=(int(1e5),))

    if use_logx:
        x = np.log(x+1e-10)
    x = x[:, np.newaxis]

    if force_thres is not None:
        thres_ = np.log(force_thres) if use_logx else force_thres
    else:
        clf = GaussianMixture(n_components=2, means_init=[[-5], [0.]], n_init=5)
        clf.fit(x)
        x_tmp = np.linspace(x.min(), x.max(), 1000)
    
        pdf1 = multivariate_normal.pdf(x_tmp, clf.means_[0],
                                       clf.covariances_[0]) * clf.weights_[0]
        pdf2 = multivariate_normal.pdf(x_tmp, clf.means_[1],
                                       clf.covariances_[1]) * clf.weights_[1]
    
        if clf.means_[0, 0] < clf.means_[1, 0]:
            diff = pdf1 < pdf2
        else:
            diff = pdf1 > pdf2
    
        thres_ = x_tmp[np.where(diff)[0][0]]

    thres = np.exp(thres_) if use_logx else thres_

    if visualize:
        bins = np.linspace(x.min(), x.max(), 100)
        fig = plt.figure(figsize=(3, 3))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
        ax.hist(x[:, 0], bins=bins, density=True)
        if force_thres is None:
            pdf = pdf1 + pdf2
            ax.plot(x_tmp, pdf)
        ax.plot([thres_, thres_], [0, 1])

        if use_logx:
            x = np.exp(x)
            thres_ = np.exp(thres_)
            bins = np.linspace(x.min(), x.max(), 100)
            fig = plt.figure(figsize=(3, 3))
            ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
            ax.hist(x[:, 0], bins=bins, density=True)
            ax.plot([thres_, thres_], [0, 1])
            # ax.set_ylim([0, 1])

    return thres


def plot_pn2kc_claw_stats(dir, x_key, loop_key=None, dynamic_thres=False, select_dict = None, thres = THRES, ax_args=None):
    wglos = tools.load_pickle(dir, 'w_glo')
    xrange = wglos[0].shape[0]
    zero_claws = []
    mean_claws = []

    for i, wglo in enumerate(wglos):
        # dynamically infer threshold after training
        if dynamic_thres:
            thres = infer_threshold(wglo)
            print(thres)
        else:
            thres = thres
        sparsity = np.count_nonzero(wglo > thres, axis=0)

        y, _ = np.histogram(sparsity, bins=xrange, range=[0,xrange], density=True)
        zero_claws.append(np.sum(sparsity == 0)/ sparsity.size)
        mean_claws.append(np.mean(sparsity[sparsity != 0]))

    ylim = [1, 15]
    yticks = [1, 5, 7, 9, 15]

    dirs = tools._get_alldirs(dir, model=False, sort=True)
    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        setattr(config, 'mean_claw', mean_claws[i])
        setattr(config, 'zero_claw', zero_claws[i])
        tools.save_config(config, d)

    yticks_mean = [0, 3, 7, 15, 20]
    yticks_zero = [0., .5, 1]
    ax_args_0 = {'ylim':ylim,'yticks':yticks}
    ax_args_1 = {'ylim': [0, 1], 'yticks': [0, .5, 1]}
    if ax_args:
        ax_args_0.update(ax_args)
        ax_args_1.update(ax_args)
    sa.plot_results(dir, x_key=x_key, y_key='mean_claw', yticks = yticks_mean, loop_key=loop_key,
                    select_dict= select_dict,
                    ax_args= ax_args_0,
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65))

    sa.plot_results(dir, x_key=x_key, y_key='zero_claw', yticks = yticks_zero, loop_key=loop_key,
                    select_dict=select_dict,
                    ax_args= ax_args_1,
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65))

def image_pn2kc_parameters(dir, dynamic_thres=False):
    def _rank(coor):
        rank = rankdata(coor,'dense')-1
        vals, counts = np.unique(coor, return_counts=True)
        vals = [int(val) if val >= 1 else val for val in vals.tolist()]
        return rank, vals, counts

    def _image(path, xkey, ykey, zkey, zticks):
        res = tools.load_all_results(path)
        x_coor = res[xkey]
        y_coor = res[ykey]
        z_coor = res[zkey]
        x_rank, xs, x_counts = _rank(x_coor)
        y_rank, ys, y_counts = _rank(y_coor)
        image = np.zeros((np.max(x_counts), np.max(y_counts)))
        image[x_rank, y_rank] = z_coor

        rect = [0.15, 0.15, 0.65, 0.65]
        rect_cb = [0.82, 0.15, 0.02, 0.65]

        fig = plt.figure(figsize=(2.6, 2.6))
        ax = fig.add_axes(rect)
        cm = 'jet'
        if zkey == 'mean_claw':
            cm = plt.cm.get_cmap(cm, zticks[-1]-zticks[0])
        im = ax.imshow(image, cmap=cm, vmin=zticks[0], vmax=zticks[-1], interpolation='none')
        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.tick_params('both', length=0)
        ax.set_xticks(range(x_counts[0]))
        ax.set_yticks(range(y_counts[0]))
        ax.set_xticklabels(xs)
        ax.set_yticklabels(ys)

        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax)
        if zkey == 'mean_claw':
            cb.set_ticks([x+.5 for x in zticks])
            cb.set_ticklabels(zticks)
        else:
            cb.set_ticks(zticks)
        cb.outline.set_linewidth(0.5)
        cb.set_label(nicename(zkey), fontsize=7, labelpad=5)
        plt.tick_params(axis='both', which='major', labelsize=7)
        cb.ax.tick_params('both',length=0)
        plt.axis('tight')
        _easy_save(path, '_' + nicename(zkey), pdf=False)


    wglos = tools.load_pickle(dir, 'w_glo')
    xrange = wglos[0].shape[0]
    zero_claws = []
    mean_claws = []
    for wglo in wglos:
        # dynamically infer threshold after training
        thres = infer_threshold(wglo) if dynamic_thres else THRES
        sparsity = np.count_nonzero(wglo > thres, axis=0)
        y, _ = np.histogram(sparsity, bins=xrange, range=[0,xrange], density=True)
        zero_claws.append(y[0])
        mean_claws.append(np.mean(sparsity))

    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        setattr(config, 'mean_claw', mean_claws[i])
        setattr(config, 'zero_claw', zero_claws[i])
        tools.save_config(config, d)

    _image(dir, xkey='kc_loss_alpha', ykey='kc_loss_beta', zkey='val_acc', zticks=[0, .5, 1])
    _image(dir, xkey='kc_loss_alpha', ykey='kc_loss_beta', zkey='zero_claw', zticks=[0, .5, 1])
    _image(dir, xkey='kc_loss_alpha', ykey='kc_loss_beta', zkey='mean_claw', zticks=[0, 4, 7, 10, 11])


def plot_weight_distribution_per_kc(path, xrange=15, loopkey=None):
    '''
    Plots the distribution of sorted PN2KC weights
    Assumptions thus far:
    indices are no loss, loss, sparse and fixed
    sparse and fixed data is at index 2
    :param path:
    :return:
    '''

    def _plot(means, stds, thres):
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

        for i, (mean, std) in enumerate(zip(means, stds)):
            x = np.arange(0, mean.size)
            if np.mean(std) > .001:
                plt.plot(x, mean)
                plt.fill_between(x, mean - std, mean + std, alpha=.5)
            else:
                plt.step(x, mean)

        plt.plot([0, xrange], [thres, thres], '--', color='gray')
        ax.legend(legend)

        ax.set_xlabel('Connections from PNs, Sorted')
        ax.set_ylabel('Connection Weight')
        xticks = np.array([1, 5, 7, 9, 11, xrange])
        yticks = np.arange(0, yrange, .25)
        ax.set_xticks(xticks - 1)
        ax.set_xticklabels(xticks)
        ax.set_yticks(yticks)
        plt.ylim([-.01, yrange])
        plt.xlim([0, xrange])
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        _easy_save(path, str='_weights_per_kc', pdf=True)

    if loopkey is None:
        legend = ['Trained, no loss', 'Trained, with loss', 'Fixed']
    else:
        res = tools.load_all_results(path)
        legend = res[loopkey]
    wglos = tools.load_pickle(path, 'w_glo')
    means = []
    stds = []
    yrange = .6
    for wglo in wglos:
        wglo[np.isnan(wglo)] = 0
        sorted_wglo = np.sort(wglo, axis=0)
        sorted_wglo = np.flip(sorted_wglo, axis=0)
        mean = np.mean(sorted_wglo, axis=1)
        std = np.std(sorted_wglo, axis=1)
        means.append(mean)
        stds.append(std)
    # TODO: figure out how to use dynamic threshold here
    _plot(means, stds, THRES)


def plot_distribution(dir, xrange=1.0, log=False):
    save_name = dir.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path, exist_ok=True)
    if tools._islikemodeldir(dir):
        dirs = [dir]
    else:
        dirs = tools._get_alldirs(dir, model=True, sort=True)

    titles = ['Before Training', 'After Training']

    for i, d in enumerate(dirs):
        print('Analyzing results from: ' + str(d))
        try:
            wglo = tools.load_pickle(os.path.join(d, 'epoch'), 'w_glo')
        except KeyError:
            wglo = tools.load_pickle(os.path.join(d, 'epoch'), 'w_kc')
        for j in [0, -1]:
            w = wglo[j]
            w[np.isnan(w)] = 0
            distribution = w.flatten()
            save_name = os.path.join(path, 'distribution_' + str(i) + '_' + str(j))
            print(save_name)

            if j == 0:
                cutoff = 0
            else:
                cutoff = infer_threshold(distribution)

            if log == False:
                _plot_distribution(distribution, save_name, cutoff = cutoff,
                                   title=titles[j], xrange= xrange, yrange=5000)
            else:
                if j == 0:
                    approximate=False
                else:
                    approximate=True
                _plot_log_distribution(distribution, save_name, cutoff = cutoff,
                                   title=titles[j], xrange= xrange, yrange=5000, approximate=approximate)


def compute_sparsity(d, epoch, dynamic_thres=False, visualize=False,
                     thres=THRES):
    print('Analyzing results from: ' + str(d))
    try:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_glo')
    except KeyError:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_kc')
    w = wglos[epoch]
    return _compute_sparsity(w, dynamic_thres, visualize, thres)


def compute_sparsity_allepochs(d, dynamic_thres=False, visualize=False,
                     thres=THRES):
    print('Analyzing results from: ' + str(d))
    try:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_glo')
    except KeyError:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_kc')

    sparsity_list = list()
    for i, w in enumerate(wglos):
        _dynamic_thres = dynamic_thres if i > 0 else 0.1
        sparsity = _compute_sparsity(w, _dynamic_thres, visualize, thres)
        sparsity_list.append(sparsity)
    return sparsity_list


def _compute_sparsity(w, dynamic_thres=False, visualize=False, thres=THRES):
    w[np.isnan(w)] = 0

    # dynamically infer threshold after training
    if dynamic_thres is False:
        thres = thres
    elif dynamic_thres == True:
        thres = None
    else:
        thres = dynamic_thres
    thres = infer_threshold(w, visualize=visualize, force_thres=thres)
    print('thres=', str(thres))

    sparsity = np.count_nonzero(w > thres, axis=0)
    return sparsity


def plot_sparsity(dir, dynamic_thres=False, visualize=False, thres=THRES,
                  epochs=None):
    save_name = dir.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path,exist_ok=True)

    if tools._islikemodeldir(dir):
        dirs = [dir]
    else:
        dirs = tools._get_alldirs(dir, model=True, sort=True)

    if epochs is None:
        epochs = [-1]  # analyze the first and the last
        titles = ['Before Training', 'After Training']
        yrange = [1, 0.5]
    else:
        titles = ['Epoch {:d}'.format(e) for e in epochs]
        yrange = [0.5]*len(epochs)

    for i, d in enumerate(dirs):
        for j, epoch in enumerate(epochs):
            sparsity = compute_sparsity(d, epoch, dynamic_thres=dynamic_thres,
                                        visualize=visualize, thres=thres)
            save_name = os.path.join(path, 'sparsity_' + str(i) + '_' + str(j))
            _plot_sparsity(sparsity, save_name, title= titles[j], yrange= yrange[j])
            print(sparsity.mean())


def _plot_sparsity(data, savename, title, xrange=50, yrange=.5):
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(data, bins=xrange, range=[0, xrange], density=True, align='left')
    plt.plot([7, 7], [0, yrange], '--', color='gray')
    ax.set_xlabel('PN inputs per KC')
    ax.set_ylabel('Fraction of KCs')
    name = title
    ax.set_title(name)

    xticks = [1, 7, 15, 25, 50]
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(0, yrange, 3))
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    print(savename)
    plt.savefig(savename + '.png', dpi=500)
    plt.savefig(savename + '.pdf', transparent=True)

def _plot_log_distribution(data, savename, title, xrange, yrange, cutoff = 0, approximate=True):
    # if visualize:
    #     bins = np.linspace(x.min(), x.max(), 100)
    #     fig = plt.figure(figsize=(3, 3))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    #     ax.hist(x[:, 0], bins=bins, density=True)
    #     if force_thres is None:
    #         pdf = pdf1 + pdf2
    #         ax.plot(x_tmp, pdf)
    #     ax.plot([thres_, thres_], [0, 1])
    #
    #     if use_logx:
    #         x = np.exp(x)
    #         thres_ = np.exp(thres_)
    #         bins = np.linspace(x.min(), x.max(), 100)
    #         fig = plt.figure(figsize=(3, 3))
    #         ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    #         ax.hist(x[:, 0], bins=bins, density=True)
    #         ax.plot([thres_, thres_], [0, 1])
    #         # ax.set_ylim([0, 1])



    # y = np.log(data)
    x = np.log(data + 1e-10)
    cutoff = np.log(cutoff)
    xrange = np.log(xrange)

    xticks = ['$10^{-6}$','$10^{-4}$', '.01', '1']
    xticks_log = np.log([1e-6, 1e-4, 1e-2, 1])

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.28, 0.25, 0.6, 0.6])
    if approximate:
        plt.hist(x, bins=50, range = [-12, 3], density=True)
    else:
        weights = np.ones_like(x) / float(len(x))
        plt.hist(x, bins=50, range = [-12, 3], weights=weights)
    ax.set_xlabel('PN to KC Weight')
    ax.set_ylabel('Distribution of Connections')
    name = title
    ax.set_title(name)
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
    ax.plot([cutoff, cutoff], [0, plt.ylim()[1]], '--', color='gray', linewidth=1)

    if approximate:
        x = np.asarray(data).flatten()
        x = np.log(x + 1e-10)
        x = x[:, np.newaxis]
        clf = GaussianMixture(n_components=2)
        clf.fit(x)
        x_tmp = np.linspace(x.min(), x.max(), 1000)

        pdf1 = multivariate_normal.pdf(x_tmp, clf.means_[0],
                                           clf.covariances_[0]) * clf.weights_[0]
        pdf2 = multivariate_normal.pdf(x_tmp, clf.means_[1],
                                           clf.covariances_[1]) * clf.weights_[1]

        ax.plot(x_tmp, pdf1, linestyle='--', linewidth=1, alpha = 1)
        ax.plot(x_tmp, pdf2, linestyle='--', linewidth=1, alpha = 1)
        # ax.plot(x_tmp, pdf1 + pdf2, color='black', linewidth=1, alpha = .5)

    plt.savefig(savename + '_log.png', dpi=500)
    plt.savefig(savename + '_log.pdf', transparent=True)


def _plot_distribution(data, savename, title, xrange, yrange, broken_axis=True, cutoff = 0):
    fig = plt.figure(figsize=(2, 1.5))
    if not broken_axis:
        ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
        plt.hist(data, bins=50, range=[0, xrange], density=False)
        ax.set_xlabel('PN to KC Weight')
        ax.set_ylabel('Number of Connections')
        name = title
        ax.set_title(name)

        xticks = [0, .2, .4, .6, .8, 1]
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        yticks = [0, 1000, 2000, 3000, 4000, 5000]
        yticklabels = ['0', '1K', '2K', '3K', '4K', '>100K']
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        plt.ylim([0, yrange])
        plt.xlim([0, xrange])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.plot([cutoff, cutoff], [0, yrange], '--', color='gray')

    else:
        ax = fig.add_axes([0.25, 0.25, 0.7, 0.45])
        ax2 = fig.add_axes([0.25, 0.75, 0.7, 0.1])
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

        name = title
        ax2.set_title(name)

        ax.set_xlabel('PN to KC Weight')
        ax.set_ylabel('Number of Connections')
        xticks = np.arange(0, xrange + 0.01, .5)
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
        yticks = [0, 2500, 5000]
        yticklabels = ['0', '2.5K', '5K']
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(0, yrange)  # most of the data

        ax2.set_yticks([np.max(n)])
        ax2.set_yticklabels(['{:d}K'.format(int(np.max(n)/1000))])
        ax2.set_ylim(0.9 * np.max(n), 1.1 * np.max(n))  # outliers only
        ax.plot([cutoff, cutoff], [0, yrange], '--', color='gray')
        ax2.plot([cutoff, cutoff], ax2.get_ylim(), '--', color='gray')

        plt.savefig(savename + '.png', dpi=500)
        plt.savefig(savename + '.pdf', transparent=True)


def plot_sparsity_acrossepochs(path, dynamic_thres=False):
    import tools
    config = tools.load_config(path)
    res = tools.load_all_results(os.path.join(rootpath, 'files', 'longtrain'),
                                 argLast=False)
    n_epoch = len(os.listdir(os.path.join(path, 'epoch')))
    epochs = np.arange(n_epoch)

    sparsity_list = compute_sparsity_allepochs(
        path, dynamic_thres=True, visualize=False)

    Ks = list()
    for i, epoch in enumerate(epochs):
        sparsity = sparsity_list[i]
        Ks.append(sparsity[sparsity > 0].mean())
        print(epoch, sparsity[sparsity > 0].mean())
        print(epoch, len(sparsity[sparsity > 0]))

    Ks = np.array(Ks)

    savename = os.path.join(figpath, 'sparsity_acrossepochs')

    final_K = Ks[-1]
    Ks[Ks > final_K * 1.5] = np.nan

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.28, 0.25, 0.6, 0.6])
    ax.plot(epochs, Ks)
    plt.xlabel('Epochs')
    plt.ylabel('K')
    plt.ylim([0, np.nanmax(Ks) * 1.1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.savefig(savename + '.png', dpi=500)
    plt.savefig(savename + '.pdf', transparent=True)

    # TODO: Temporarily dumped here
    wglos = tools.load_pickle(os.path.join(path, 'epoch'), 'w_glo')
    bglos = tools.load_pickle(os.path.join(path, 'epoch'),
                              'model/layer2/bias:0')

    # =============================================================================
    # thres = analysis_pn2kc_training.infer_threshold(
    #         wglos[190], visualize=True, force_thres=None)
    # =============================================================================

    # =============================================================================
    # epoch = 190
    # w = wglos[epoch]
    # b = bglos[epoch]
    # use_logx = True
    # force_thres = np.exp(-2.5)
    # visualize = True
    # from scipy.stats import multivariate_normal
    # from sklearn.mixture import GaussianMixture
    #
    # x = np.array(w)
    # ratio = np.max(x, axis=0) / np.median(x, axis=0)
    # # heuristic that works well for N=50-500, can plot hist of ratio
    # ind = ratio > 15
    # print('Valid units {:d}/{:d}'.format(np.sum(ind), len(ind)))
    # x = x[:, ind]  # select expansion layer neurons
    # b = b[ind]
    #
    # bs = list()
    # for epoch in epochs:
    #     w = wglos[epoch]
    #     b = bglos[epoch]
    #
    #     x = np.array(w)
    #     ratio = np.max(x, axis=0) / np.median(x, axis=0)
    #     # heuristic that works well for N=50-500, can plot hist of ratio
    #     ind = ratio > 15
    #
    #     bs.append(b[ind].mean())
    #
    # plt.plot(epochs, bs)
    # =============================================================================

    # =============================================================================
    # x = x.flatten()
    # if use_logx:
    #     x = np.log(x+1e-10)
    # x = x[:, np.newaxis]
    #
    # if force_thres is not None:
    #     thres_ = np.log(force_thres) if use_logx else force_thres
    # else:
    #     clf = GaussianMixture(n_components=2)
    #     clf.fit(x)
    #     x_tmp = np.linspace(x.min(), x.max(), 1000)
    #
    #     pdf1 = multivariate_normal.pdf(x_tmp, clf.means_[0],
    #                                    clf.covariances_[0]) * clf.weights_[0]
    #     pdf2 = multivariate_normal.pdf(x_tmp, clf.means_[1],
    #                                    clf.covariances_[1]) * clf.weights_[1]
    #
    #     if clf.means_[0, 0] < clf.means_[1, 0]:
    #         diff = pdf1 < pdf2
    #     else:
    #         diff = pdf1 > pdf2
    #
    #     thres_ = x_tmp[np.where(diff)[0][0]]
    #
    # thres = np.exp(thres_) if use_logx else thres_
    #
    # if visualize:
    #     bins = np.linspace(x.min(), x.max(), 100)
    #     fig = plt.figure(figsize=(5, 5))
    #     ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    #     ax.hist(x[:, 0], bins=bins, density=True)
    #     if force_thres is None:
    #         pdf = pdf1 + pdf2
    #         ax.plot(x_tmp, pdf)
    #     ax.plot([thres_, thres_], [0, 1])
    #     plt.ylim([0, 0.4])
    #
    #     if use_logx:
    #         x = np.exp(x)
    #         thres_ = np.exp(thres_)
    #         bins = np.linspace(x.min(), x.max(), 100)
    #         fig = plt.figure(figsize=(3, 3))
    #         ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    #         ax.hist(x[:, 0], bins=bins, density=True)
    #         ax.plot([thres_, thres_], [0, 1])
    #         # ax.set_ylim([0, 1])
    #
    # sparsity = np.count_nonzero(w > thres, axis=0)
    # print('K', sparsity[sparsity>0].mean())
    # =============================================================================

    # =============================================================================
    # # Analyze activity
    # from standard.analysis import load_activity
    # glo_in, glo_out, kc_out, results = load_activity(path)
    # print('Activity sparsity', np.mean(kc_out[:, ind] > 0.))
    # =============================================================================

# if __name__ == '__main__':
#     dir = "../files/train_KC_claws"
#     plot_sparsity(dir)
#     plot_distribution(dir)


