"""Analyze the trained models."""

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import dict_methods
from scipy.signal import savgol_filter

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
from settings import seqcmap

figpath = os.path.join(rootpath, 'figures')


def _get_ax_args(xkey, ykey, n_pn=50):
    unique_n_pns = np.unique(n_pn)
    if len(unique_n_pns) == 1:
        n_pn = unique_n_pns[0]
    else:
        n_pn = 50
    ax_args = {}
    if ykey in ['K_inferred', 'sparsity_inferred', 'K', 'sparsity', 'K_smart']:
        if n_pn == 50:
            if xkey in ['kc_norm']:
                ax_args['ylim'] = [0, 30]
                ax_args['yticks'] = [1, 10, 20, 30]
            else:
                ax_args['ylim'] = [0, 20]
                ax_args['yticks'] = [1, 5, 10, 15, 20]
        else:
            ax_args['ylim'] = [0, int(0.5*n_pn)]
    elif ykey in ['val_acc', 'glo_score', 'coding_level']:
        ax_args['ylim'] = [0, 1.05]
        ax_args['yticks'] = [0, .5, 1]
    elif ykey == 'log_train_loss':
        ax_args['ylim'] = [-2, 2]
        ax_args['yticks'] = [-2, -1, 0, 1, 2]

    rect = (0.3, 0.35, 0.5, 0.55)

    if xkey == 'kc_inputs':
        ax_args['xticks'] = [3, 7, 15, 30, 40, 50]
    if ykey == 'kc_inputs':
        ax_args['yticks'] = [3, 7, 15, 30, 40, 50]

    if 'xticks' in ax_args.keys():
        ax_args['xticks'] = np.array(ax_args['xticks'])

    if 'yticks' in ax_args.keys():
        ax_args['yticks'] = np.array(ax_args['yticks'])

    return rect, ax_args


def _infer_plot_xy_axargs(X, Y):
    xlims = list()
    ylims = list()
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        ypeak = np.percentile(y, 95)
        # first value from right higher than ypeak * 0.01
        xlims.append(x[-np.where(np.array(y[::-1]) > ypeak*0.01)[0][0]])
        ylims.append(ypeak*1.2)

    xlim = np.max(xlims)
    ylim = np.max(ylims)
    return {'ylim': [0, ylim], 'xlim': [0, xlim]}


def plot_xy(save_path, xkey, ykey, select_dict=None, legend_key=None,
            ax_args=None, res=None, figsize=None):
    if not save_path:
        return

    def _plot_xy(xkey, ykey):
        ys = res[ykey]
        xs = res[xkey]

        if xkey in ['lin_bins', 'log_bins']:
            xs = [(x[1:] + x[:-1]) / 2 for x in xs]

        if ykey == 'lin_hist':
            new_ys = []
            for i in range(len(xs)):
                x = xs[i]
                y = ys[i]
                # Plot density by dividing by binsize
                bin_size = x[1] - x[0]  # must have equal bin size
                y = y / bin_size
                if res['kc_prune_weak_weights'][i]:
                    y[0] = 0  # Ignore the pruned ones
                # Smoothing
                window_lenth = int(0.02 / bin_size / 2) * 2 + 1
                y = savgol_filter(
                    y, window_length=window_lenth, polyorder=0)
                new_ys.append(y)
            ys = new_ys

        ax_args_ = _infer_plot_xy_axargs(xs, ys)
        if ax_args is not None:
            ax_args_.update(ax_args)

        _figsize = figsize or (1.5, 1.5)
        rect = [0.15, 0.25, 0.8, 0.6]
        fig = plt.figure(figsize=_figsize)
        ax = fig.add_axes(rect, **ax_args_)

        colors = [seqcmap(x) for x in np.linspace(0, 1, len(xs))]
        for x, y, c in zip(xs, ys, colors):
            ax.plot(x, y, alpha=1, color=c, linewidth=1)

        if legend_key is not None:
            legends = res[legend_key]
            legends = [nicename(l, mode=legend_key) for l in legends]
            ncol = 1 if len(legends) < 4 else 2
            ax.legend(legends, fontsize=6, frameon=False, ncol=ncol,
                      loc='best')
            ax.set_title(nicename(legend_key), fontsize=7)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if ykey == 'lin_hist':
            ax.set_yticks([])

        figname = xkey + '_' + ykey
        if legend_key:
            figname = figname + '_' + legend_key
        if select_dict:
            for k, v in select_dict.items():
                figname += k + '_' + str(v) + '_'
        tools.save_fig(save_path, figname)

    if res is None:
        res = tools.load_all_results(save_path, argLast=True)
    if select_dict is not None:
        res = dict_methods.filter(res, select_dict)

    if legend_key:
        # get rid of duplicates
        values = res[legend_key]
        if np.any(values == None):
            values[values == None] = 'None'
        _, ixs = np.unique(values, return_index=True)
        for k in [legend_key, xkey, ykey]:
            res[k] = res[k][ixs]
    _plot_xy(xkey, ykey)


def plot_progress(save_path, select_dict=None, alpha=1, exclude_dict=None,
                  legend_key=None, epoch_range=None, ykeys=None,
                  ax_args=None, show_cleanpn2kc=True, show_ylabel=True,
                  ):
    """Plot progress through training.

    Args:
        save_path: str or list of strs

    """
    if ykeys is None:
        ykeys = ['log_val_loss', 'log_train_loss', 'val_loss',
                 'train_loss', 'val_acc', 'glo_score']

    if not save_path:
        return

    if isinstance(ykeys, str):
        ykeys = [ykeys]
    ny = len(ykeys)
    res = tools.load_all_results(save_path, argLast=False,
                                 select_dict=select_dict,
                                 exclude_dict=exclude_dict)

    figsize = [2.0, 1.2 + 0.7 * (ny-1)]
    if not show_ylabel:
        figsize[0] = figsize[0] - 0.3
    fig, axs = plt.subplots(ny, 1, figsize=figsize, sharex='all')
    xkey = 'epoch'
    for i, ykey in enumerate(ykeys):
        rect, ax_args_ = _get_ax_args(xkey, ykey, n_pn=res['N_PN'])
        if ax_args:
            ax_args_.update(ax_args)

        rect = [0.3, 0.3, 0.65, 0.5]

        # ax = fig.add_axes(rect, **ax_args_)
        if ny == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.update(ax_args_)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ys = res[ykey]
        xs = res[xkey]
        clean_pn2kc = res['clean_pn2kc']

        if legend_key:
            # Sort by legend key
            ind_sort = np.argsort(res[legend_key])
            xs, ys = xs[ind_sort], ys[ind_sort]
            legends = res[legend_key][ind_sort]
            clean_pn2kc = clean_pn2kc[ind_sort]

        colors = [seqcmap(x) for x in np.linspace(0, 1, len(xs))]

        for j, (x, y, c) in enumerate(zip(xs, ys, colors)):
            if epoch_range:
                x, y = x[epoch_range[0]:epoch_range[1]], y[epoch_range[0]:epoch_range[1]]
            if show_cleanpn2kc and ('K' in ykey) and (not clean_pn2kc[j]):
                pass  # Not showing at all
                # ax.plot(x, y, '--', alpha=alpha*0.5, color=c, linewidth=1)
            else:
                ax.plot(x, y, alpha=alpha, color=c, linewidth=1)

        if legend_key is not None and i == 0:
            legends = [nicename(l, mode=legend_key) for l in legends]
            ncol = 1 if len(legends) < 3 else 2
            ax.legend(legends, fontsize=6, frameon=False, ncol=ncol,
                      loc='best')
            ax.set_title(nicename(legend_key), fontsize=7)

        if i != ny - 1:
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel(nicename(xkey))

        if show_ylabel:
            ax.set_ylabel(nicename(ykey))
        else:
            ax.set_yticklabels([])

        if epoch_range:
            ax.set_xlim([epoch_range[0], epoch_range[1]])
        else:
            ax.set_xlim([-1, res[xkey][0,-1]])

    plt.tight_layout()

    figname = '_' + '_'.join(ykeys)
    if select_dict:
        for k, v in select_dict.items():
            figname += k + '_' + str(v) + '_'

    if legend_key:
        figname += '_' + legend_key

    if epoch_range:
        figname += '_epoch_range_' + str(epoch_range[1])

    tools.save_fig(save_path, figname)


def plot_weights(modeldir, var_name=None, average=False, vlim=None,
                 zoomin=False, **kwargs):
    """Plot weights of a model."""
    if var_name is None:
        var_name = ['w_or', 'w_orn', 'w_combined', 'w_glo',
                    'w_step2to3', 'w_step3to4']
    elif isinstance(var_name, str):
        var_name = [var_name]
    for v in var_name:
        _plot_weights(modeldir, v, average=average, vlim=vlim,
                      zoomin=zoomin, **kwargs)


def _load_sorted_pickle(modeldir):
    """Sort neurons in weight matrices of var_dict."""
    var_dict = tools.load_pickle(modeldir)
    # Each weight matrix in this is (Input neuron, Output neuron)
    config = tools.load_config(modeldir)
    new_var_dict = dict()
    if 'w_or' in var_dict.keys():
        # First sort ORNs
        w = var_dict['w_or']  # OR-ORN weight
        ind_sort_orn = np.argsort(np.argmax(w, axis=0))
        new_var_dict['w_or'] = w[:, ind_sort_orn]
        new_var_dict['w_orn'] = var_dict['w_orn'][ind_sort_orn, :]
    else:
        w = var_dict['w_orn']  # ORN-PN weight
        if config.N_ORN_DUPLICATION > 1:
            n_orn = config.N_ORN * config.N_ORN_DUPLICATION
            assert w.shape[0] == n_orn
            # Reorder ORNs so the neurons of same type are adjacent
            ind_sort_orn = np.concatenate(
                [range(i, n_orn, config.N_ORN) for i in range(config.N_ORN)])
            new_var_dict['w_orn'] = w[ind_sort_orn, :]
        else:
            new_var_dict['w_orn'] = w

    if 'w_step2to3' in var_dict.keys():
        # Only used for RNN, Step=3 experiment
        # Next sort PNs
        w = new_var_dict['w_orn']  # ORN-PN weight
        ind_sort_pn = np.argsort(np.argmax(w, axis=0))
        new_var_dict['w_orn'] = w[:, ind_sort_pn]
        new_var_dict['w_step2to3'] = var_dict['w_step2to3'][ind_sort_pn, :]

        # Then sort copy layer
        w = new_var_dict['w_step2to3']  # PN-copy layer weight
        ind_sort_copy = np.argsort(np.argmax(w, axis=0))
        new_var_dict['w_step2to3'] = w[:, ind_sort_copy]
        new_var_dict['w_step3to4'] = var_dict['w_step3to4'][ind_sort_copy, :]
        new_var_dict['w_glo'] = np.dot(new_var_dict['w_step2to3'],
                                       new_var_dict['w_step3to4'])
    else:
        # Next sort PNs
        w = new_var_dict['w_orn']  # ORN-PN weight
        ind_sort_pn = np.argsort(np.argmax(w, axis=0))
        new_var_dict['w_orn'] = w[:, ind_sort_pn]
        new_var_dict['w_glo'] = var_dict['w_glo'][ind_sort_pn, :]

    # KCs are not sorted
    if 'w_or' in var_dict.keys():
        new_var_dict['w_combined'] = np.dot(
            new_var_dict['w_or'], new_var_dict['w_orn'])
    return new_var_dict


def _plot_weights(modeldir, var_name='w_orn', average=False,
                  vlim=None, binarized=False, title_keys=None, zoomin=False,
                  ax_args=None):
    """Plot weights."""
    # Load network at the end of training
    var_dict = _load_sorted_pickle(modeldir)

    try:
        w_plot = var_dict[var_name]
    except KeyError:
        # Weight doesn't exist, return
        return
    print('Plotting ' + var_name + ' from ' + modeldir)

    if average:
        # Should only be used for w_orn
        # Using repeat mode because _load_sorted_pickle changed w_orn
        w_orn_by_pn = tools.reshape_worn(w_plot, 50, mode='repeat')
        w_plot = w_orn_by_pn.mean(axis=0)

    if var_name == 'w_glo':
        w_plot = w_plot[:, :20]

    if binarized and var_name == 'w_glo':
        log = tools.load_log(modeldir)
        w_plot = (w_plot > log['thres_inferred'][-1]) * 1.0

    # w_max = np.max(abs(w_plot))
    w_max = np.percentile(abs(w_plot), 99)
    _w_max = (np.round(w_max, decimals=1) if w_max > .1
              else np.round(w_max, decimals=2))
    positive_w = np.min(w_plot) > -1e-6  # all weights positive

    if not vlim:
        if positive_w:
            vlim = [0, _w_max]
        else:
            vlim = [-_w_max, _w_max]

    if not zoomin:
        figsize = (1.7, 1.7)
        rect = [0.15, 0.15, 0.6, 0.6]
        rect_cb = [0.77, 0.15, 0.02, 0.6]
    else:
        figsize = (5.0, 5.0)  # Matplotlib wouldn't render properly if small
        rect = [0.05, 0.05, 0.9, 0.9]
        n_zoom = 10
        w_plot = w_plot[:n_zoom, :][:, :n_zoom]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)

    cmap = plt.get_cmap('RdBu_r')
    if positive_w:
        cmap = tools.truncate_colormap(cmap, 0.5, 1.0)

    im = ax.imshow(w_plot.T, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                   interpolation='nearest',
                   extent=(-0.5, w_plot.shape[0]-0.5, w_plot.shape[1]-0.5,
                           -0.5))

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)

    if zoomin:
        ax.set_xticks([])
        ax.set_yticks([])
        # Minor ticks
        ax.set_xticks(np.arange(-.5, n_zoom), minor=True)
        ax.set_yticks(np.arange(-.5, n_zoom), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.tick_params('both', length=0)
        # Gridlines based on minor ticks
        ax.grid(which='minor', color='w', linestyle='-', linewidth=3)
    else:
        labelpad = -5
        if var_name == 'w_orn':
            y_label, x_label = 'To PNs', 'From ORNs'
        elif var_name == 'w_or':
            y_label, x_label = 'To ORNs', 'From ORs'
        elif var_name == 'w_glo':
            y_label, x_label = 'To KCs', 'From PNs'
        elif var_name == 'w_combined':
            y_label, x_label = 'To PNs', 'From ORs'
        else:
            y_label, x_label = '', ''
        ax.set_ylabel(y_label, labelpad=labelpad)
        ax.set_xlabel(x_label, labelpad=labelpad)
        title = tools.nicename(var_name)
        if title_keys is not None:
            config = tools.load_config(modeldir)
            if isinstance(title_keys, str):
                title_keys = [title_keys]
            for title_key in title_keys:
                v = getattr(config, title_key)
                title += '\n' + tools.nicename(
                    title_key) + ':' + tools.nicename(v, 'lr')
        ax.set_title(title, fontsize=7)
        ax.set_xticks([0, w_plot.shape[0] - 1, w_plot.shape[0]])
        ax.set_xticklabels(['1', str(w_plot.shape[0]), ''])
        ax.set_yticks([0, w_plot.shape[1] - 1, w_plot.shape[1]])
        ax.set_yticklabels(['1', str(w_plot.shape[1]), ''])
        ax.tick_params('both', length=0)

        if ax_args is not None:
            ax.update(ax_args)
            if 'title' in ax_args.keys():
                ax.set_title(ax_args['title'], fontsize=7)

        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax, ticks=vlim)
        cb.outline.set_linewidth(0.5)
        cb.set_label('Weight', labelpad=-7)
        plt.tick_params(axis='both', which='major')
        plt.axis('tight')

    var_name = var_name.replace('/', '_')
    var_name = var_name.replace(':', '_')
    figname = '_' + var_name + '_' + tools.get_model_name(modeldir)
    if zoomin:
        figname = figname + '_zoom'
    tools.save_fig(tools.get_experiment_name(modeldir), figname)


def plot_results(path, xkey, ykey, loop_key=None, select_dict=None,
                 logx=None, logy=False, figsize=None, ax_args=None,
                 plot_args=None, ax_box=None, res=None, string='',
                 plot_actual_value=True, show_ylabel=True,
                 show_cleanpn2kc=True):
    """Plot results for varying parameters experiments.

    Args:
        path: str, model save path
        xkey: str, key for the x-axis variable
        ykey: str, key for the y-axis variable
        loop_key: str, key for the value to loop around
        select_dict: dict, dict of parameters to select
        logx: bool, if True, use log x-axis
        logy: bool, if True, use log x-axis
    """
    if isinstance(ykey, str):
        ykeys = [ykey]
    else:
        ykeys = ykey
    ny = len(ykeys)

    if res is None:
        res = tools.load_all_results(path, select_dict=select_dict)

    tmp = res[xkey][0]
    xkey_is_string = isinstance(tmp, str) or tmp is None

    if plot_args is None:
        plot_args = {}

    # X-axis should be shared for all curves in this plot, precomputed
    if logx is None:
        logx = xkey in ['lr', 'meta_lr', 'meta_update_lr',
                        'N_KC', 'N_PN', 'initial_pn2kc',
                        'kc_prune_threshold', 'N_ORN_DUPLICATION',
                        'n_trueclass', 'n_trueclass_ratio',
                        'meta_num_samples_per_class']

    # Unique sorted xkey values
    xvals = sorted(set(res[xkey]))
    if xkey_is_string:
        x_plot = np.arange(len(xvals))
    else:
        x_plot = np.log(np.array(xvals)) if logx else np.array(xvals)

    if figsize is None:
        figsize = [1.5, 1.2 + 0.7 * (ny - 1)]
        if not show_ylabel:
            figsize[0] -= 0.3
        if xkey in ['lr', 'N_KC', 'N_PN', 'spread_orn_activity',
                    'n_trueclass_ratio', 'kc_recinh_coeff', 'PN-KC Connections']:
            figsize[0] += 0.3
        if xkey in ['orn_corr']:
            figsize[0] += 1.0
        if xkey == 'spread_orn_activity':
            figsize[0] += 1.0

    def _plot(_ykey, ind=None, label=None, color=None,
              plot_actual_value=False):
        """

        Args:
             _ykey: str, y key for line
             ind: optinonal bool array of entries to select
        """
        yvals = list()
        yval_alls = list()
        clean_pn2kc = list()
        for xval in xvals:
            _ind = res[xkey] == xval
            if ind is not None:
                _ind = _ind * ind
            yval_tmp = res[_ykey][_ind]
            yvals.append(np.median(yval_tmp))
            yval_alls.append(yval_tmp)

            clean_pn2kc_tmp = res['clean_pn2kc'][_ind]
            clean_pn2kc.append(all(clean_pn2kc_tmp))
        yvals = np.array(yvals)
        clean_pn2kc = np.array(clean_pn2kc)

        y_plot = np.log(yvals) if logy else yvals

        _y_plot = y_plot.copy()
        if show_cleanpn2kc and 'K' in _ykey:
            _y_plot[~clean_pn2kc] = np.nan
        ax.plot(x_plot, _y_plot, 'o-', markersize=3, label=label,
                color=color, **plot_args)

        if plot_actual_value:
            for i, (x, y) in enumerate(zip(x_plot, y_plot)):
                if y > ax.get_ylim()[-1]:
                    continue
                if show_cleanpn2kc and ('K' in _ykey) and (not clean_pn2kc[i]):
                    continue
                if _ykey in ['val_acc', 'glo_score', 'coding_level']:
                    ytext = '{:0.2f}'.format(y)
                else:
                    ytext = '{:0.1f}'.format(y)
                ax.text(x, y, ytext, fontsize=6,
                        horizontalalignment='center',
                        verticalalignment='bottom')

    fig, axs = plt.subplots(ny, 1, figsize=figsize, sharex='all')
    for i, ykey in enumerate(ykeys):
        # Default ax_args and other values, based on x and y keys
        rect, ax_args_ = _get_ax_args(xkey, ykey, n_pn=res['N_PN'])
        if ax_args:
            ax_args_.update(ax_args)
        if ax_box is not None:
            rect = ax_box
        ax = axs[i] if ny > 1 else axs
        ax.update(ax_args_)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if loop_key:
            loop_vals = np.unique(res[loop_key])
            colors = [seqcmap(x) for x in np.linspace(0, 1, len(loop_vals))]
            for loop_val, color in zip(loop_vals, colors):
                ind = res[loop_key] == loop_val
                _plot(ykey, ind=ind, label=nicename(loop_val, mode=loop_key),
                      color=color)
        else:
            _plot(ykey, plot_actual_value=plot_actual_value)

        if 'xticks' in ax_args_.keys():
            xticks = ax_args_['xticks']
            xticklabels = [nicename(x, mode=xkey) for x in xticks]
            if logx:
                xticks = np.log(xticks)
        else:
            xticks = x_plot
            xticklabels = [nicename(x, mode=xkey) for x in xvals]
        ax.set_xticks(xticks)
        if i == ny - 1:
            if xkey in ['kc_norm_pre', 'kc_norm_post', 'kc_norm',
                        'training_type']:
                ax.set_xticklabels(xticklabels, rotation=15, ha="right")
            else:
                ax.set_xticklabels(xticklabels)
            ax.set_xlabel(nicename(xkey))
        else:
            ax.tick_params(labelbottom=False)

            # ax.set_xticks(xticks)
        # if not xkey_is_string:
        #     x_span = xticks[-1] - xticks[0]
        #     ax.set_xlim([xticks[0]-x_span*0.05, xticks[-1]+x_span*0.05])
        # ax.set_xticklabels(xticklabels)

        if 'yticks' in ax_args_.keys():
            yticks = ax_args_['yticks']
            if logy:
                ax.set_yticks(np.log(yticks))
                ax.set_yticklabels(yticks)
            else:
                ax.set_yticks(yticks)
            if not show_ylabel:
                ax.set_yticklabels(['' for _ in yticks])
        else:
            plt.locator_params(axis='y', nbins=3)
        if show_ylabel:
            ax.set_ylabel(nicename(ykey))

        if xkey == 'kc_inputs':
            ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
        elif xkey == 'N_PN':
            ax.plot([np.log(50), np.log(50)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')
        elif xkey == 'N_KC':
            ax.plot([np.log(2500), np.log(2500)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')

        if loop_key and i == 0:
            ncol = 1 if len(loop_vals) < 4 else 1

            if xkey == 'spread_orn_activity':
                l = ax.legend(loc='upper left', fontsize=7, frameon=False,
                              ncol=1, bbox_to_anchor=(1., 1.))
            else:
                l = ax.legend(loc='best', fontsize=7, frameon=False, ncol=ncol)
            l.set_title(nicename(loop_key))
    plt.tight_layout()

    figname = '_' + '_'.join(ykeys) + '_vs_' + xkey
    if loop_key:
        figname += '_vary' + loop_key
    if select_dict:
        for k, v in select_dict.items():
            if isinstance(v, list):
                v = [x.rsplit('/',1)[-1] for x in v]
                v = str('__'.join(v))
            else:
                v = str(v)
            figname += k + '_' + v + '__'
    figname += string
    tools.save_fig(path, figname)


if __name__ == '__main__':
    pass