import os
import sys
import pickle
import glob
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from scipy.signal import find_peaks

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import save_fig
import dict_methods
from standard.analysis_weight import infer_threshold


figpath = os.path.join(rootpath, 'figures')
THRES = 0.1


def _set_colormap(nbins):
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]
    cmap_name = 'my_list'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=nbins)
    return cm


def check_single_peak_obsolete(bins, hist, threshold):
    """Check if an array of histogram has double peaks.

    Args:
        bins: (n_networks, n_histogram_points)
        hist: (n_networks, n_histogram_points)
        threshold: (n_networks)

    Returns:
        peak_ind: bool array (n_networks), True if single peak
    """
    # TODO: This has a problem if lin_bins doesn't cover the second peak
    peak_ind = np.zeros_like(threshold).astype(np.bool)
    for i, thres in enumerate(threshold):
        # find location of threshold
        ind_thres = np.where(bins[i, :-1] > thres)[0][0]
        # with=10, heuristics
        hist_ = hist[i][ind_thres:]
        peaks, properties = find_peaks(hist_, width=20)
        # check if there's an additional peak undetected
        peak_loc = np.argmax(hist_)
        peak_ind[i] = len(peaks) == 1 and abs(peaks[0] - peak_loc) < 5
    return peak_ind


def has_nobadkc(modeldir, bad_kc_threshold=0.2):
    """Check if model has too many bad KCs."""
    log = tools.load_log(modeldir)
    # After training, bad KC proportion should lower 'bad_kc_threshold'
    return log['bad_KC'][-1] < bad_kc_threshold


def filter_modeldirs_badkc(modeldirs, bad_kc_threshold=0.2):
    """Filter model dirs with too many bad KCs."""
    return [d for d in modeldirs if has_nobadkc(d, bad_kc_threshold)]


def has_singlepeak(modeldir, peak_threshold=0.1):
    """Check if model has a single peak."""
    # TODO: Use this method throughout to replace similar methods
    log = tools.load_log(modeldir)
    config = tools.load_config(modeldir)
    if config.kc_prune_weak_weights:
        thres = config.kc_prune_threshold
    else:
        thres = log['thres_inferred'][-1]  # last epoch
    if len(log['lin_bins'].shape) == 1:
        bins = log['lin_bins'][:-1]
    else:
        bins = log['lin_bins'][-1, :-1]
    bin_size = bins[1] - bins[0]
    hist = log['lin_hist'][-1]  # last epoch
    # log['lin_bins'] shape (nbin+1), log['lin_hist'] shape (n_epoch, nbin)
    ind_thres = np.argsort(np.abs(bins - thres))[0]
    ind_grace = int(0.01 / bin_size)  # grace distance to start find peak
    hist_abovethres = hist[ind_thres + ind_grace:]
    ind_peak = np.argmax(hist_abovethres)
    # Value at threshold and at peak
    thres_value = hist_abovethres[0]
    peak_value = hist_abovethres[ind_peak]
    if ind_peak * bin_size <= peak_threshold or peak_value < 1.3 * thres_value:
        # peak should be at least 'peak_threshold' away from threshold
        return False
    else:
        return True


def filter_modeldirs_badpeak(modeldirs, peak_threshold=0.1):
    """Filter model dirs without a strong second peak."""
    return [d for d in modeldirs if has_singlepeak(d, peak_threshold)]


def filter_modeldirs(modeldirs, exclude_badkc=False, exclude_badpeak=False):
    """Select model directories.

    Args:
        modeldirs: list of model directories
        exclude_badkc: bool, if True, exclude models with too many bad KCs
        exclude_badpeak: bool, if True, exclude models with bad peaks

    Return:
        modeldirs: list of filtered model directories
    """
    print('Analyzing {} model directories'.format(len(modeldirs)))
    if exclude_badkc:
        modeldirs = filter_modeldirs_badkc(modeldirs)
        print('{} remain after filtering bad kcs'.format(len(modeldirs)))
    if exclude_badpeak:
        modeldirs = filter_modeldirs_badpeak(modeldirs)
        print('{} remain after filtering bad peaks'.format(len(modeldirs)))
    return modeldirs


def do_everything(path, filter_peaks=False, redo=False, range=2, select_dict=None):
    def _get_K_obsolete(res):
        # GRY: Not sure what this function is doing
        n_model, n_epoch = res['sparsity'].shape[:2]
        Ks = np.zeros((n_model, n_epoch))
        bad_KC = np.zeros((n_model, n_epoch))
        for i in range(n_model):
            if res['kc_prune_weak_weights'][i]:
                Ks[i] = res['K'][i]
            else:
                Ks[i] = res['K_inferred'][i]

                # sparsity = res['sparsity'][i, j]
                # Ks[i, j] = sparsity[sparsity>0].mean()
                # bad_KC[i,j] = np.sum(sparsity==0)/sparsity.size
        res['K_inferred'] = Ks
        res['bad_KC'] = bad_KC

    d = os.path.join(path)
    files = glob.glob(d)
    res = defaultdict(list)
    for f in files:
        temp = tools.load_all_results(f, argLast=False, select_dict=select_dict)
        dict_methods.chain_defaultdicts(res, temp)

    if redo:
        wglos = tools.load_pickles(path, 'w_glo')
        for i, wglo in enumerate(wglos):
            w = wglo.flatten()
            hist, bins = np.histogram(w, bins=1000, range=[0, range])
            res['lin_bins'][i] = bins
            res['lin_hist'][i][-1,:] = hist #hack
        # _get_K(res)

    badkc_ind = res['bad_KC'][:, -1] < 0.2
    acc_ind = res['train_acc'][:, -1] > 0.5
    if filter_peaks:
        peak_ind = check_single_peak(res['lin_bins'],
                                     res['lin_hist'][:, -1, :],  # last epoch
                                     res['kc_prune_threshold'])
    else:
        peak_ind = np.ones_like(acc_ind)
    ind = badkc_ind * acc_ind * peak_ind
    for k, v in res.items():
        res[k] = v[ind]

    for k in res['lin_bins']:
        res['lin_bins_'].append(k[:-1])
    for k in res['lin_hist']:
        res['lin_hist_'].append(savgol_filter(k[-1], window_length=21, polyorder=0))
    for k, v in res.items():
        res[k] = np.array(res[k])
    return res


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


def plot_sparsity(modeldir, epoch=None, dynamic_thres=True,
                  visualize=False, thres=THRES, xrange=50, plot=True):
    model_name = tools.get_model_name(modeldir)

    if epoch is not None and epoch != -1:
        modeldir = tools.get_modeldirs(os.path.join(modeldir, 'epoch'))[epoch]

    w = tools.load_pickles(modeldir, 'w_glo')[0]
    sparsity, thres = _compute_sparsity(w, dynamic_thres, visualize, thres)

    if plot:
        if epoch is not None:
            string = '_epoch_' + str(epoch)
        else:
            string = ''

        save_path = os.path.join(figpath, tools.get_experiment_name(modeldir))
        save_name = os.path.join(save_path, '_' + model_name + '_sparsity' + string)
        _plot_sparsity(sparsity, save_name, xrange=xrange)
    return sparsity


def _plot_sparsity(data, savename, xrange=50, yrange=None):
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])
    plt.hist(data, bins=xrange, range=[0, xrange], density=True, align='left')
    # plt.plot([7, 7], [0, yrange], '--', color='gray')
    ax.set_xlabel('PN inputs per KC')
    ax.set_ylabel('Fraction of KCs')

    if yrange is None:
        hist, _ = np.histogram(data, bins=xrange, range=[0, xrange],
                               density=True)
        vmax = np.max(hist)
        if vmax > 0.5:
            yrange = 1
        elif vmax > 0.25:
            yrange = 0.5
        else:
            yrange = 0.25

    xticks = [1, 7, 15, 25, 50]
    ax.set_xticks(xticks)
    ax.set_yticks(np.linspace(0, yrange, 3))
    plt.ylim([0, yrange])
    plt.xlim([-1, xrange])
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
    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.25, 0.25, 0.7, 0.6])

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

    save_path = os.path.join(figpath, tools.get_experiment_name(modeldir))
    save_name = os.path.join(save_path, '_' + model_name + '_')

    _plot_distribution(
        distribution, save_name + 'distribution' + string,
        thres=thres, xrange=xrange, yrange=5000, **kwargs)
    _plot_log_distribution(
        distribution, save_name + 'log_distribution' + string,
        thres=thres, res_fit=res_fit, **kwargs)


def _plot_log_distribution(data, savename, thres=0, res_fit=None, **kwargs):
    x = np.log(data + 1e-10)

    xticks = ['$10^{-6}$','$10^{-4}$', '.01', '1']
    xticks_log = np.log([1e-6, 1e-4, 1e-2, 1])

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.28, 0.25, 0.6, 0.6])
    if res_fit is not None:
        plt.hist(x, bins=50, range=[-12, 3], density=True)
    else:
        weights = np.ones_like(x) / float(len(x))
        plt.hist(x, bins=50, range=[-12, 3], weights=weights)

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
        if thres is not None:
            ax.plot([thres, thres], [0, yrange], '--', color='gray')
    else:
        if xrange is None:
            xrange = np.max(data)
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
        if thres is not None:
            ax.plot([thres, thres], [0, yrange], '--', color='gray')
            ax2.plot([thres, thres], ax2.get_ylim(), '--', color='gray')

        split = os.path.split(savename)
        tools.save_fig(split[0], split[1])


def plot_log_distribution_movie(modeldir):
    log = tools.load_log(modeldir)

    xticks = ['$10^{-6}$','$10^{-4}$', '.01', '1']
    xticks_log = np.log([1e-6, 1e-4, 1e-2, 1])

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.28, 0.25, 0.6, 0.6])

    xdata, ydata = log['log_bins'][:-1], log['log_hist'][0]

    fig = plt.figure(figsize=(2, 1.5))
    ax = fig.add_axes([0.28, 0.25, 0.6, 0.6])
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
