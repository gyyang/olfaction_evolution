"""Analyze the trained models."""

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import dict_methods

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename, save_fig
import task
from standard.analysis_pn2kc_training import check_single_peak

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

figpath = os.path.join(rootpath, 'figures')


def _get_ax_args(xkey, ykey, n_pn=50):
    ax_args = {}
    if ykey in ['K_inferred', 'sparsity_inferred', 'K', 'sparsity', 'K_smart']:
        if n_pn == 50:
            ax_args['ylim'] = [0, 20]
            ax_args['yticks'] = [0, 3, 7, 10, 15, 20]
        else:
            ax_args['ylim'] = [0, int(0.5*n_pn ** 0.8)]
    elif ykey in ['val_acc', 'glo_score']:
        ax_args['ylim'] = [0, 1.05]
        ax_args['yticks'] = [0, .25, .5, .75, 1]
    elif ykey == 'train_logloss':
        ax_args['ylim'] = [-2, 2]
        ax_args['yticks'] = [-2, -1, 0, 1, 2]

    if xkey == 'lr':
        rect = (0.2, 0.25, 0.75, 0.65)
    else:
        rect = (0.27, 0.25, 0.65, 0.65)

    if xkey == 'kc_inputs':
        ax_args['xticks'] = [3, 7, 15, 30, 40, 50]
    if ykey == 'kc_inputs':
        ax_args['yticks'] = [3, 7, 15, 30, 40, 50]

    if 'xticks' in ax_args.keys():
        ax_args['xticks'] = np.array(ax_args['xticks'])

    if 'yticks' in ax_args.keys():
        ax_args['yticks'] = np.array(ax_args['yticks'])

    return rect, ax_args


def plot_xy(save_path, xkey, ykey, select_dict=None, legend_key=None, ax_args=None, log=None):
    def _plot_progress(xkey, ykey):
        ax_args_ = {}
        if ax_args is None:
            if ykey == 'lin_hist_':
                ax_args_ = {'ylim': [0, 500]}

        figsize = (2.5, 2)
        rect = [0.3, 0.3, 0.65, 0.5]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect, **ax_args_)

        ys = log[ykey]
        xs = log[xkey]

        colors = [cm.cool(x) for x in np.linspace(0, 1, len(xs))]

        for x, y, c in zip(xs, ys, colors):
            ax.plot(x, y, alpha= 1, color = c, linewidth=1)

        if legend_key is not None:
            legends = log[legend_key]
            legends = [nicename(l, mode=legend_key) for l in legends]
            ax.legend(legends, fontsize=7, frameon=False, ncol= 2, loc='best')
            ax.set_title(nicename(legend_key), fontsize=7)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        if ykey == 'val_acc' and log[ykey].shape[0] == 1:
            plt.title('Final accuracy {:0.3f}'.format(log[ykey][0,-1]), fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        figname = '_' + ykey
        if select_dict:
            for k, v in select_dict.items():
                figname += k + '_' + str(v) + '_'
        save_fig(save_path, figname)

    if log is None:
        log = tools.load_all_results(save_path, argLast=False)
    if select_dict is not None:
        log = dict_methods.filter(log, select_dict)
    if legend_key:
        # get rid of duplicates
        values = log[legend_key]
        if np.any(values == None):
            values[values == None] = 'None'
        _, ixs = np.unique(values, return_index=True)
        for k, v in log.items():
            log[k] = log[k][ixs]
    _plot_progress(xkey, ykey)


def plot_progress(save_path, select_dict=None, alpha=1, exclude_dict=None,
                  legend_key=None, epoch_range=None, ykeys=None, ax_args=None):
    """Plot progress through training.
        Fixed to allow for multiple plots
    """

    log = tools.load_all_results(save_path, argLast=False)
    if select_dict is not None:
        log = dict_methods.filter(log, select_dict)
    if exclude_dict is not None:
        log = dict_methods.exclude(log, exclude_dict)

    # get rid of duplicates
    if legend_key:
        values = log[legend_key]
        if np.any(values == None):
            values[values == None] = 'None'
        _, ixs = np.unique(values, return_index=True)
        for k, v in log.items():
            log[k] = log[k][ixs]

    def _plot_progress(xkey, ykey):
        rect, ax_args_ = _get_ax_args(xkey, ykey, n_pn=log['N_PN'][0])
        if ax_args:
            ax_args_.update(ax_args)

        figsize = (2.5, 2)
        rect = [0.3, 0.3, 0.65, 0.5]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect, **ax_args_)

        ys = log[ykey]
        xs = log[xkey]

        colors = [cm.cool(x) for x in np.linspace(0, 1, len(xs))]

        for x, y, c in zip(xs, ys, colors):
            if epoch_range:
                x, y = x[epoch_range[0]:epoch_range[1]], y[epoch_range[0]:epoch_range[1]]
            ax.plot(x, y, alpha=alpha, color = c, linewidth=1)

        if legend_key is not None:
            # ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 1.2), fontsize=4)
            legends = log[legend_key]
            legends = [nicename(l, mode=legend_key) for l in legends]
            ax.legend(legends, fontsize=7, frameon=False, ncol= 2, loc='best')
            plt.title(nicename(legend_key), fontsize=7)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        if ykey == 'val_acc' and log[ykey].shape[0] == 1:
            plt.title('Final accuracy {:0.3f}'.format(log[ykey][0,-1]), fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        if epoch_range:
            ax.set_xlim([epoch_range[0], epoch_range[1]])
        else:
            ax.set_xlim([-1, log[xkey][0,-1]])

        figname = '_' + ykey
        if select_dict:
            for k, v in select_dict.items():
                figname += k + '_' + str(v) + '_'

        if epoch_range:
            figname += '_epoch_range_' + str(epoch_range[1])
        save_fig(save_path, figname)

    if ykeys is None:
        ykeys = ['val_logloss', 'train_logloss', 'val_loss',
                 'train_loss', 'val_acc', 'glo_score']

    if isinstance(ykeys, str):
        ykeys = [ykeys]

    for plot_var in ykeys:
        _plot_progress('epoch', plot_var)


def plot_weights(path, var_name='w_orn', sort_axis=1, average=False,
                 vlim=None, positive_cmap=True):
    """Plot weights.

    Currently this plots OR2ORN, ORN2PN, and OR2PN
    """
    # Load network at the end of training
    model_dir = os.path.join(path, 'model.pkl')
    print('Plotting ' + var_name + ' from ' + model_dir)
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        if var_name == 'w_combined':
            w_plot = np.dot(var_dict['w_or'], var_dict['w_orn'])
        else:
            w_plot = var_dict[var_name]


    # if not hasattr(config, 'receptor_layer') or config.receptor_layer == False:
    #     if config.replicate_orn_with_tiling:
    #         weight = np.reshape(
    #             weight, (config.N_ORN_DUPLICATION, config.N_ORN, config.N_PN))
    #         weight = np.swapaxes(weight, 0, 1)
    #         weight = np.reshape(weight, (-1, config.N_PN))

    if average:
        w_orn_by_pn = tools._reshape_worn(w_plot, 50)
        w_plot = w_orn_by_pn.mean(axis=0)
    # Sort for visualization
    if sort_axis == 0:
        ind_max = np.argmax(w_plot, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_plot[:, ind_sort]
    elif sort_axis == 1:
        ind_max = np.argmax(w_plot, axis=1)
        ind_sort = np.argsort(ind_max)
        w_plot = w_plot[ind_sort, :]
    else:
        pass

    if var_name == 'w_glo':
        w_plot = w_plot[:, :20]

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)

    w_max = np.max(abs(w_plot))
    if not vlim:
        vlim = [0, np.round(w_max, decimals=1) if w_max > .1 else np.round(w_max, decimals=2)]

    cmap = plt.get_cmap('RdBu_r')
    if positive_cmap:
        cmap = tools.truncate_colormap(cmap, 0.5, 1.0)

    im = ax.imshow(w_plot.T, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                   interpolation='none')

    if var_name == 'w_orn':
        plt.title('ORN-PN connectivity after training', fontsize=7)
        ax.set_ylabel('To PNs', labelpad=-5)
        ax.set_xlabel('From ORNs', labelpad=-5)
    elif var_name == 'w_or':
        plt.title('OR-ORN expression array after training', fontsize=7)
        ax.set_ylabel('To ORNs', labelpad=-5)
        ax.set_xlabel('From ORs', labelpad=-5)
    elif var_name == 'w_glo':
        plt.title('PN-KC connectivity after training', fontsize=7)
        ax.set_ylabel('To KCs', labelpad=-5)
        ax.set_xlabel('from PNs', labelpad=-5)
    elif var_name == 'w_combined':
        plt.title('OR-PN combined connectivity', fontsize=7)
        ax.set_ylabel('To PNs', labelpad=-5)
        ax.set_xlabel('From ORs', labelpad=-5)
    else:
        print('unknown variable name for weight matrix: {}'.format(var_name))

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_yticks([0, w_plot.shape[1]])
    ax.set_xticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=vlim)
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    save_fig(os.path.split(path)[0], '_' + var_name + '_' + os.path.split(path)[1])


    # Plot distribution of various connections
    # keys = var_dict.keys()
    # keys = ['model/layer1/bias:0', 'model/layer2/bias:0']
    # for key in keys:
    #     fig = plt.figure(figsize=(2, 2))
    #     plt.hist(var_dict[key].flatten())
    #     plt.title(key)


def load_activity(save_path, lesion_kwargs=None):
    '''
    Loads model activity from tensorflow
    :param save_path:
    :return:
    '''
    import tensorflow as tf
    from model import SingleLayerModel, FullModel, NormalizedMLP
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    config.label_type = 'sparse'

    # Load dataset
    data_dir = rootpath + config.data_dir[1:]  # this is a hack as well
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, data_dir)

    tf.reset_default_graph()
    if config.model == 'full':
        CurrentModel = FullModel
    elif config.model == 'singlelayer':
        CurrentModel = SingleLayerModel
    elif config.model == 'normmlp':
        CurrentModel = NormalizedMLP
    else:
        raise ValueError('Unknown model type ' + str(config.model))
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    # model.save_path = rootpath + model.save_path[1:]
    model.save_path = save_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        if lesion_kwargs:
            model.lesion_units(**lesion_kwargs)

        # Validation
        glo_out, glo_in, kc_in, kc_out, logits = sess.run(
            [model.glo, model.glo_in, model.kc_in, model.kc, model.logits],
            {val_x_ph: val_x, val_y_ph: val_y})
        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    # to_show = 100
    # plt.subplot(1, 2, 1)
    # if val_y.ndim == 1:
    #     mat = np.zeros_like(logits[:to_show])
    #     mat[np.arange(to_show), val_y[:to_show]] = 1
    #     plt.imshow(mat)
    # else:
    #     plt.imshow(val_y[:to_show,:])
    # plt.subplot(1,2,2)
    # plt.imshow(logits[:to_show,:])
    # plt.show()
    return glo_in, glo_out, kc_in, kc_out, results

def plot_activity(save_path):
    glo_in, glo_out, kc_in, kc_out, results = load_activity(save_path)
    save_name = save_path.split('/')[-1]
    plt.figure()
    plt.hist(glo_out.flatten(), bins=100)
    plt.title('Glo activity distribution')
    plt.savefig(os.path.join(figpath, save_name + '_pn_activity.pdf'), transparent=True)
    plt.show()
    
    plt.figure()
    plt.hist(kc_out.flatten(), bins=100)
    plt.title('KC activity distribution')
    plt.savefig(os.path.join(figpath, save_name + '_kc_activity.pdf'), transparent=True)
    plt.show()


def plot_results(path, xkey, ykey, loop_key=None, select_dict=None,
                 logx=None, logy=False, figsize=None, ax_args=None,
                 plot_args=None, ax_box=None, sort=True, res=None, string='',
                 plot_actual_value=False, filter_peaks=False):
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
    if res is None:
        res = tools.load_all_results(path)

    if select_dict is not None:
        res = dict_methods.filter(res, select_dict)

    if plot_args is None:
        plot_args = {}

    # Sort by xkey
    if sort:
        ind_sort = np.argsort(res[xkey])
        for key, val in res.items():
            res[key] = val[ind_sort]

    if logx is None:
        logx = xkey in ['lr', 'N_KC', 'initial_pn2kc', 'kc_prune_threshold',
                         'N_ORN_DUPLICATION']

    if figsize is None:
        if xkey == 'lr':
            figsize = (4.5, 1.5)
        else:
            figsize = (1.5, 1.5)

    def _plot_results(ykey):
        # Default ax_args and other values, based on x and y keys
        rect, ax_args_ = _get_ax_args(xkey, ykey, n_pn=res['N_PN'][0])
        if ax_args:
            ax_args_.update(ax_args)
        if ax_box is not None:
            rect = ax_box

        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect, **ax_args_)
        if loop_key:
            loop_vals = np.unique(res[loop_key])
            colors = [cm.cool(x) for x in np.linspace(0, 1, len(loop_vals))]
            for loop_val, color in zip(loop_vals, colors):
                ind = res[loop_key] == loop_val
                x_plot = res[xkey][ind]
                y_plot = res[ykey][ind]
                if logx:
                    x_plot = np.log(x_plot)
                if logy:
                    y_plot = np.log(y_plot)
                # x_plot = [str(x).rsplit('/', 1)[-1] for x in x_plot]
                ax.plot(x_plot, y_plot, 'o-', markersize=3, color=color,
                        label=nicename(loop_val, mode=loop_key), **plot_args)
        else:
            x_plot = res[xkey]
            y_plot = res[ykey]

            # Get rid of duplicates
            _, ix = np.unique(x_plot, return_index=True)
            x_plot = x_plot[ix]
            y_plot = y_plot[ix]

            if filter_peaks:
                peak_ind = check_single_peak(res['lin_bins'][ix],
                                             res['lin_hist'][ix],
                                             res['kc_prune_threshold'][ix])
                x_plot = x_plot[peak_ind]
                y_plot = y_plot[peak_ind]

            if logx:
                x_plot = np.log(x_plot)
            if logy:
                y_plot = np.log(y_plot)
            ax.plot(x_plot, y_plot, 'o-', markersize=3, **plot_args)
            if plot_actual_value:
                for x, y in zip(x_plot, y_plot):
                    if y > ax.get_ylim()[-1]:
                        continue
                    if ykey in ['val_acc', 'glo_score']:
                        ytext = '{:0.2f}'.format(y)
                    else:
                        ytext = '{:0.1f}'.format(y)
                    ax.text(x, y, ytext,
                            horizontalalignment='center',
                            verticalalignment='bottom')

        if 'xticks' in ax_args_.keys():
            xticks = ax_args_['xticks']
        else:
            xticks = np.unique(res[xkey])

        xticklabels = [nicename(x, mode=xkey) for x in xticks]
        xticks = np.log(xticks) if logx else xticks

        ax.set_xticks(xticks)
        x_span = xticks[-1] - xticks[0]
        ax.set_xlim([xticks[0]-x_span*0.05, xticks[-1]+x_span*0.05])
        ax.set_xticklabels(xticklabels)

        if 'yticks' in ax_args_.keys():
            yticks = ax_args_['yticks']
        else:
            yticks = np.unique(res[ykey])

        if logy:
            ax.set_yticks(np.log(yticks))
            ax.set_yticklabels(yticks)
        else:
            ax.set_yticks(yticks)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))

        if xkey == 'kc_inputs':
            ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
        elif xkey == 'N_PN':
            ax.plot([np.log(50), np.log(50)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')
        elif xkey == 'N_KC':
            ax.plot([np.log(2500), np.log(2500)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')

        if loop_key:
            l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5), fontsize= 7,
                          frameon=False, ncol=2)
            l.set_title(nicename(loop_key))

        figname = '_' + ykey + '_vs_' + xkey
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

        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        save_fig(path, figname)

    if isinstance(ykey, str):
        ykey = [ykey]

    for ykey_tmp in ykey:
        _plot_results(ykey_tmp)


if __name__ == '__main__':
    pass