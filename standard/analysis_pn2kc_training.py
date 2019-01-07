import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from tools import nicename
import utils
from standard.analysis import _easy_save
import standard.analysis as sa
from scipy.stats import rankdata
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import kurtosis

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # TODO: This is hacky, should be fixed
mpl.rcParams['font.size'] = 7
figpath = os.path.join(rootpath, 'figures')
thres = 0.05

def _set_colormap(nbins):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm

def plot_pn2kc_claw_stats(dir, x_key, loop_key=None):
    wglos = tools.load_pickle(dir, 'w_glo')
    xrange = wglos[0].shape[0]
    zero_claws = []
    mean_claws = []

    for wglo in wglos:
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

    yticks_mean = [0, 3, 7, 15, 20]
    yticks_zero = [0., .5, 1]
    sa.plot_results(dir, x_key=x_key, y_key='mean_claw', yticks = yticks_mean, loop_key=loop_key)
    sa.plot_results(dir, x_key=x_key, y_key='zero_claw', yticks = yticks_zero, loop_key=loop_key)

def image_pn2kc_parameters(dir):
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

    def _plot(means, stds):
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

        _easy_save(path, str='_weights_per_kc', pdf=False)

    if loopkey is None:
        legend = ['Trained, no loss', 'Trained, loss', 'Sparse and fixed']
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
    _plot(means, stds)

def plot_sparsity(dir):
    def _plot_sparsity(data, savename, title, xrange=50, yrange= .5):
        fig = plt.figure(figsize=(2.5, 2))
        ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
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
        plt.xlim([0, xrange])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.savefig(savename + '.png', dpi=500)

    save_name = dir.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path,exist_ok=True)
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    titles = ['Before Training', 'After Training']
    yrange = [1, 0.5]
    for i, d in enumerate(dirs):
        wglo = tools.load_pickle(os.path.join(d,'epoch'), 'w_glo')
        wglo = [wglo[0]] + [wglo[-1]]
        for j, w in enumerate(wglo):
            w[np.isnan(w)] = 0
            sparsity = np.count_nonzero(w > thres, axis=0)
            save_name = os.path.join(path, 'sparsity_' + str(i) + '_' + str(j))
            _plot_sparsity(sparsity, save_name, title= titles[j], yrange= yrange[j])

def plot_distribution(dir):
    save_name = dir.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path,exist_ok=True)
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    titles = ['Before Training', 'After Training']

    def _plot_distribution(data, savename, title, xrange, yrange, broken_axis=True):
        fig = plt.figure(figsize=(3, 2))
        if not broken_axis:
            ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
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

        else:
            ax = fig.add_axes([0.2, 0.2, 0.7, 0.5])
            ax2 = fig.add_axes([0.2, 0.75, 0.7, 0.1])
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
            xticks = [0, .2, .4, .6, .8, 1]
            ax.set_xticks(xticks)
            ax.set_xticklabels([str(x) for x in xticks])
            yticks = [0, 1000, 2000, 3000, 4000, 5000]
            yticklabels = ['0', '1K', '2K', '3K', '4K', '5K']
            ax.set_yticks(yticks)
            ax.set_yticklabels(yticklabels)
            ax.set_ylim(0, yrange)  # most of the data

            ax2.set_yticks([np.max(n)])
            ax2.set_yticklabels(['{:d}K'.format(int(np.max(n)/1000))])
            ax2.set_ylim(0.9 * np.max(n), 1.1 * np.max(n))  # outliers only

        plt.savefig(savename + '.png', dpi=500)

    for i, d in enumerate(dirs):
        wglo = tools.load_pickle(os.path.join(d,'epoch'), 'w_glo')
        for j in [0, -1]:
            w = wglo[j]
            w[np.isnan(w)] = 0
            distribution = w.flatten()
            save_name = os.path.join(path, 'distribution_' + str(i) + '_' + str(j))
            _plot_distribution(distribution, save_name,
                               title=titles[j], xrange=1.0, yrange=5000)


# if __name__ == '__main__':
#     dir = "../files/train_KC_claws"
#     plot_sparsity(dir)
#     plot_distribution(dir)


