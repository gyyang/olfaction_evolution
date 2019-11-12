import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from tools import nicename
from tools import save_fig
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
mpl.rcParams['mathtext.fontset'] = 'stix'

figpath = os.path.join(rootpath, 'figures')
THRES = 0.1

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
    # Select neurons that receive both strong and weak connections
    # weak connections should be around median, where strong should be around max
    x = np.array(x)
    ratio = np.max(x, axis=0) / np.median(x, axis=0)
    # heuristic that works well for N=50-500, can plot hist of ratio
    ind = ratio > 15
    if np.sum(ind) > 0:
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
        clf = GaussianMixture(n_components=2, means_init=[[-5], [0.]], n_init=1)
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

        try:
            thres_ = x_tmp[np.where(diff)[0][0]]
        except IndexError:
            print('Unable to find proper threshold, revert to default')
            thres_ = np.log(THRES) if use_logx else THRES

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

def plot_distribution(dir, dir_ix, epoch=None, xrange=1.0, log=False):
    dir_folder = tools._get_alldirs(dir, model=True, sort=True)[dir_ix]
    folder = os.path.split(dir_folder)[-1]
    experiment = os.path.split(dir)[-1]

    if epoch is not None:
        dir_folder = tools._get_alldirs(os.path.join(dir_folder, 'epoch'), model=True, sort=True)[epoch]

    try:
        w = tools.load_pickle(dir_folder, 'w_glo')[0]
    except KeyError:
        w = tools.load_pickle(dir_folder, 'w_kc')[0]
    w[np.isnan(w)] = 0
    distribution = w.flatten()

    if epoch is not None:
        string = '_epoch_' + str(epoch)
    else:
        string = ''

    if epoch == 0:
        cutoff = None
        approximate = False
    else:
        cutoff = infer_threshold(distribution)
        approximate = True

    save_path = os.path.join(figpath, experiment)
    save_name = os.path.join(save_path, '_' + folder + '_distribution'  + string)
    if log == False:
        _plot_distribution(distribution, save_name, cutoff = cutoff, xrange= xrange, yrange=5000)
    else:
        _plot_log_distribution(distribution, save_name, cutoff = cutoff,
                               xrange= xrange, yrange=5000, approximate=approximate)


def compute_sparsity(d, epoch, dynamic_thres=False, visualize=False,
                     thres=THRES):
    print('compute sparsity needs to be replaced')
    try:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_glo')
    except KeyError:
        wglos = tools.load_pickle(os.path.join(d, 'epoch'), 'w_kc')
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
    thres = infer_threshold(w, visualize=visualize, force_thres=thres)
    print('thres=', str(thres))

    sparsity = np.count_nonzero(w > thres, axis=0)
    return sparsity, thres


def plot_sparsity(dir, dir_ix, epoch=None, dynamic_thres=False, visualize=False, thres=THRES, xrange = 50, plot=True):
    dir_folder = tools._get_alldirs(dir, model=True, sort=True)[dir_ix]
    folder = os.path.split(dir_folder)[-1]
    experiment = os.path.split(dir)[-1]

    if epoch is not None:
        dir_folder = tools._get_alldirs(os.path.join(dir_folder, 'epoch'), model=True, sort=True)[epoch]

    try:
        w = tools.load_pickle(dir_folder, 'w_glo')[0]
    except KeyError:
        w = tools.load_pickle(dir_folder, 'w_kc')[0]
    sparsity, thres = _compute_sparsity(w, dynamic_thres, visualize, thres)

    if plot:
        if epoch is not None:
            string = '_epoch_' + str(epoch)
        else:
            string = ''

        if epoch == 0:
            yrange = 1
        else:
            yrange = 0.5
        save_path = os.path.join(figpath, experiment)
        save_name = os.path.join(save_path, '_' + folder + '_sparsity' + string)
        _plot_sparsity(sparsity, save_name, yrange= yrange, xrange=xrange)
    return sparsity


def _plot_sparsity(data, savename, xrange=50, yrange=.5):
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(data, bins=xrange, range=[0, xrange], density=True, align='left')
    plt.plot([7, 7], [0, yrange], '--', color='gray')
    ax.set_xlabel('PN inputs per KC')
    ax.set_ylabel('Fraction of KCs')

    xticks = [1, 7, 15, 25, 50]
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(0, yrange, 3))
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange])
    plt.title(data[data>0].mean())

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    split = os.path.split(savename)
    tools.save_fig(split[0], split[1])

def _plot_log_distribution(data, savename, xrange, yrange, cutoff = 0, approximate=True):
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

    if cutoff is not None:
        cutoff = np.log(cutoff)
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

    split = os.path.split(savename)
    tools.save_fig(split[0], split[1])


def _plot_distribution(data, savename, xrange, yrange, broken_axis=True, cutoff = None):
    fig = plt.figure(figsize=(2, 1.5))
    if not broken_axis:
        ax = fig.add_axes([0.25, 0.25, 0.6, 0.6])
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
        plt.ylim([0, yrange])
        plt.xlim([0, xrange])

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if cutoff is not None:
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
        if cutoff is not None:
            ax.plot([cutoff, cutoff], [0, yrange], '--', color='gray')
            ax2.plot([cutoff, cutoff], ax2.get_ylim(), '--', color='gray')

        split = os.path.split(savename)
        tools.save_fig(split[0], split[1])

def plot_all_K(n_orns, Ks, plot_scatter=False,
               plot_box=False, plot_data=True,
               plot_fit=True, plot_angle=False,
               plot_dim=False,
               path='default'):
    from sklearn.linear_model import LinearRegression

    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    
    logKs = [np.log(K) for K in Ks]
    med_logKs = np.array([np.median(np.log(K)) for K in Ks])
    
    if plot_scatter:
        for n_orn, K, med_logK in zip(n_orns, Ks, med_logKs):
            ax.scatter(np.log([n_orn]*len(K)), np.log(K), alpha=0.01, s=3)
            ax.plot(np.log(n_orn), med_logK, '+', ms=15, color='black')
            
    
    def _pretty_box(x, positions, ax, color):
        flierprops = {'markersize': 3, 'markerfacecolor': color,
                      'markeredgecolor': 'none'}
        boxprops = {'facecolor': color, 'linewidth': 1, 'color': color}
        medianprops = {'color': color*0.5}
        whiskerprops = {'color': color}
        ax.boxplot(x, positions=positions, widths=0.06,
                   patch_artist=True, medianprops=medianprops,
                   flierprops=flierprops, boxprops=boxprops, showcaps=False,
                   whiskerprops=whiskerprops
                   )

    def _fit(x, y):
        x_fit = np.linspace(min(np.log(25), x[0]), max(np.log(1600), x[-1]), 3)
        # model = Ridge()
        model = LinearRegression()
        model.fit(x[:, np.newaxis], y)
        y_fit = model.predict(x_fit[:, np.newaxis])
        return x_fit, y_fit, model

    if plot_box:
        _pretty_box(logKs, np.log(n_orns), ax, tools.blue)
        
    if plot_fit:
        print(n_orns)
        print(np.exp(med_logKs))
        x_fit, y_fit, model = _fit(np.log(n_orns), med_logKs)
        label = r'Train $K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])
        ax.plot(x_fit, y_fit, color=tools.blue, label=label)
    
    if plot_data:
        ax.plot(np.log(1000), np.log(100), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(1000), np.log(120), '[2]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='bottom', zorder=5)
        
        ax.plot(np.log(1000), np.log(40), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(1000), np.log(32), '[3]', color=tools.darkblue,
                horizontalalignment='center', verticalalignment='top', zorder=5)
        ax.plot(np.log(50), np.log(7), 'x', color=tools.darkblue, zorder=5)
        ax.text(np.log(53), np.log(6), '[1]', color=tools.darkblue,
                horizontalalignment='left', verticalalignment='top', zorder=5)

    if plot_dim:
        from analytical.analyze_simulation_results import _load_result
        x, y = _load_result('all_value_withdim_m', v_name='dim')
        print(x, y)
        x, y = np.log(x), np.log(y)
        plt.scatter(x, y, c=tools.gray, s=5)
        x_fit, y_fit, model = _fit(x, y)
        label = r'Max Dimension $K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])
        ax.plot(x_fit, y_fit, color=tools.gray, label=label)
    
    if plot_angle:
        fname = os.path.join(rootpath, 'files', 'analytical',
                                 'control_coding_level_summary')
        summary = pickle.load(open(fname, "rb"))
        # summary: 'opt_ks', 'coding_levels', 'conf_ints', 'n_orns'
        _pretty_box(list(np.log(summary['opt_ks'].T)),
                    np.log(summary['n_orns']), ax, tools.red)
        
    if plot_angle and plot_fit:
        # x, y = np.log(n_orns)[3:], med_logKs[3:]
        x = np.log(summary['n_orns'])
        y = np.median(np.log(summary['opt_ks']), axis=0)
        x_fit, y_fit, model = _fit(x, y)
        label = r'Robust weights $K ={:0.2f} \ N^{{{:0.2f}}}$'.format(
                               np.exp(model.intercept_), model.coef_[0])
        ax.plot(x_fit, y_fit, color=tools.red, label=label)
        
    ax.legend(bbox_to_anchor=(0., 1.05), loc=2, frameon=False)
        
    x = [ 50, 100, 150, 200, 300, 400]
    y = [ 7.90428212, 10.8857362,  16.20759494,
         20.70314843, 27.50305499, 32.03561644]
    # ax.plot(np.log(x), np.log(y))
    ax.set_xlabel('Number of ORs (N)')
    ax.set_ylabel('Optimal K')
    xticks = np.array([25, 50, 100, 200, 400, 1000, 1600])
    ax.set_xticks(np.log(xticks))
    ax.set_xticklabels([str(t) for t in xticks])
    yticks = np.array([3, 10, 30, 100])
    ax.set_yticks(np.log(yticks))
    ax.set_yticklabels([str(t) for t in yticks])
    ax.set_xlim(np.log([20, 1700]))
    
    name = 'opt_k'
    if plot_scatter:
        name += '_scatter'
    if plot_box:
        name += '_box'
    if plot_data:
        name += '_data'
    if plot_fit:
        name += '_fit'
    if plot_angle:
        name += '_angle'
    save_fig(path, name)

# if __name__ == '__main__':
#     dir = "../files/train_KC_claws"
#     plot_sparsity(dir)
#     plot_distribution(dir)


