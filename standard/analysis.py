"""Analyze the trained models."""

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import dict_methods

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename
import task
from model import SingleLayerModel, FullModel, NormalizedMLP

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

figpath = os.path.join(rootpath, 'figures')
# figpath = r'C:\Users\Peter\Dropbox\olfaction_evolution\manuscript\plots'


def plot_progress(save_path, linestyles=None, select_dict=None, alpha=1,
                  legends=None, epoch_range=None, plot_vars=None, ylim = None):
    """Plot progress through training.
        Fixed to allow for multiple plots
    """

    log = tools.load_all_results(save_path, argLast=False)
    if select_dict is not None:
        log = dict_methods.filter(log, select_dict)

    def _plot_progress(xkey, ykey):
        figsize = (3, 2)
        rect = [0.3, 0.3, 0.65, 0.5]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)

        ys = log[ykey]
        xs = log[xkey]

        lstyles = ['-'] * len(xs) if linestyles is None else linestyles
        from matplotlib import cm
        colors = [cm.cool(x) for x in np.linspace(0, 1, len(xs))]

        for x, y, s, c in zip(xs, ys, lstyles, colors):
            if epoch_range:
                x, y = x[epoch_range[0]:epoch_range[1]], y[epoch_range[0]:epoch_range[1]]
            ax.plot(x, y, alpha=alpha, linestyle=s, color = c, linewidth=2)

        if legends is not None:
            # ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 1.2), fontsize=4)
            ax.legend(legends, fontsize=7, frameon=False, ncol= 2, loc='best')

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        if ykey == 'val_acc' and log[ykey].shape[0] == 1:
            plt.title('Final accuracy {:0.3f}'.format(log[ykey][0,-1]), fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # ax.xaxis.set_ticks(np.arange(0, log[xkey][0,-1]+2, 10))
        if ykey in ['glo_score', 'or_glo_score', 'combined_glo_score']:
            ax.set_ylim([0, 1.05])
            ax.yaxis.set_ticks([0, 0.5, 1.0])

        if epoch_range:
            ax.set_xlim([epoch_range[0], epoch_range[1]])
        else:
            ax.set_xlim([-1, log[xkey][0,-1]])


        if ylim is not None:
            ax.set_ylim(ylim)

        figname = '_' + ykey
        if select_dict:
            for k, v in select_dict.items():
                figname += k + '_' + str(v) + '_'

        if epoch_range:
            figname += '_epoch_range_' + str(epoch_range[1])
        _easy_save(save_path, figname)


    if plot_vars is None:
        plot_vars = ['val_logloss', 'train_logloss','val_loss','train_loss','val_acc','glo_score']

    for plot_var in plot_vars:
        _plot_progress('epoch', plot_var)


def plot_weights(path, var_name ='w_orn', sort_axis=1, dir_ix=0, average=False, vlim = None):
    """Plot weights.

    Currently this plots OR2ORN, ORN2PN, and OR2PN
    """
    # Load network at the end of training
    model_dir = os.path.join(path, 'model.pkl')
    print('Plotting ' + var_name + ' from ' + model_dir)
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
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
        w_plot = w_plot[:,:20]

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)

    max = np.max(abs(w_plot))
    if not vlim:
        vlim = [0, np.round(max, decimals=1) if max > .1 else np.round(max, decimals=2)]
    cmap = tools.get_colormap()
    # cmap = 'RdBu_r'
    im = ax.imshow(w_plot, cmap=cmap, vmin=vlim[0], vmax=vlim[1],
                   interpolation='none')


    if var_name == 'w_orn':
        plt.title('ORN-PN connectivity after training', fontsize=7)
        ax.set_xlabel('To PNs', labelpad=-5)
        ax.set_ylabel('From ORNs', labelpad=-5)
    elif var_name == 'w_or':
        plt.title('OR-ORN expression array after training', fontsize=7)
        ax.set_xlabel('To ORNs', labelpad=-5)
        ax.set_ylabel('From ORs', labelpad=-5)
    elif var_name == 'w_glo':
        plt.title('PN-KC connectivity after training', fontsize=7)
        ax.set_xlabel('To KCs', labelpad=-5)
        ax.set_ylabel('from PNs', labelpad=-5)
    elif var_name == 'w_combined':
        plt.title('OR-PN combined connectivity', fontsize=7)
        ax.set_xlabel('To PNs', labelpad=-5)
        ax.set_ylabel('From ORs', labelpad=-5)
    else:
        print('unknown variable name for weight matrix: {}'.format(var_name))

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[vlim[0], vlim[1]])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    var_name = var_name.replace('/','_')
    var_name = var_name.replace(':','_')
    save_fig(path, '_' + var_name + '_' + str(dir_ix))


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


def plot_results(path, x_key, y_key, loop_key=None, select_dict=None,
                 figsize = (2,2), ax_box = (0.25, 0.2, 0.65, 0.65),
                 ax_args={}, plot_args={}, sort = True, res = None, string = ''):
    """Plot results for varying parameters experiments.

    Args:
        path: str, model save path
        x_key: str, key for the x-axis variable
        y_key: str, key for the y-axis variable
        loop_key: str, key for the value to loop around
    """
    #TODO PW: this function is getting very messy, needs to be cleaned up
    log_plot_dict = {'N_KC': [30, 100, 1000, 10000],
                     'N_PN': [20, 50, 100, 1000],
                     'kc_loss_alpha': [.1, 1, 10, 100],
                     'kc_loss_beta': [.1, 1, 10, 100],
                     'initial_pn2kc':[.05, .1, 1],
                     'N_ORN_DUPLICATION':[1,3,10,30,100],
                     'n_trueclass':[100, 200, 500, 1000],
                     # 'val_loss':[],
                     'glo_dimensionality':[5, 50, 200, 1000]}

    plot_dict = {'kc_inputs': [3, 7, 15, 30, 40, 50]}

    if res is None:
        res = tools.load_all_results(path)

    if select_dict is not None:
        res = dict_methods.filter(res, select_dict)

    # Sort by x_key
    if sort:
        ind_sort = np.argsort(res[x_key])
        for key, val in res.items():
            res[key] = val[ind_sort]

    fig = plt.figure(figsize= figsize)
    ax = fig.add_axes(ax_box, **ax_args)
    if loop_key:
        for x in np.unique(res[loop_key]):
            ind = res[loop_key] == x
            x_plot = res[x_key][ind]
            y_plot = res[y_key][ind]
            if x_key in log_plot_dict.keys():
                x_plot = np.log(x_plot)
            if y_key in log_plot_dict.keys():
                y_plot = np.log(y_plot)
            label = str(x).rsplit('/',1)[-1]
            # x_plot = [str(x).rsplit('/', 1)[-1] for x in x_plot]
            ax.plot(x_plot, y_plot, 'o-', markersize=3, label=label, **plot_args)
    else:
        x_plot = res[x_key]
        y_plot = res[y_key]
        #get rid of duplicates
        _, ix = np.unique(x_plot, return_index=True)
        x_plot=x_plot[ix]
        y_plot=y_plot[ix]

        if x_key in log_plot_dict.keys():
            x_plot = np.log(x_plot)
        if y_key in log_plot_dict.keys():
            y_plot = np.log(y_plot)
        ax.plot(x_plot, y_plot, 'o-', markersize=3, **plot_args)

    if x_key in log_plot_dict.keys():
        xticks = np.array(log_plot_dict[x_key])
        ax.set_xticks(np.log(xticks))
        ax.set_xticklabels(xticks)
    elif x_key in plot_dict.keys():
        xticks = np.array(plot_dict[x_key])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

    if y_key in log_plot_dict.keys():
        yticks = np.array(log_plot_dict[y_key])
        ax.set_yticks(np.log(yticks))
        ax.set_yticklabels(yticks)

    ax.set_xlabel(nicename(x_key))
    ax.set_ylabel(nicename(y_key))

    if x_key == 'kc_inputs':
        ax.plot([7, 7], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color = 'gray')
    elif x_key == 'N_PN':
        ax.plot([np.log(50), np.log(50)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')
    elif x_key == 'N_KC':
        ax.plot([np.log(2500), np.log(2500)], [ax.get_ylim()[0], ax.get_ylim()[-1]], '--', color='gray')

    if loop_key:
        l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5), fontsize= 7, frameon=False)
        l.set_title(nicename(loop_key))

    figname = '_' + y_key + '_vs_' + x_key
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


if __name__ == '__main__':
    pass