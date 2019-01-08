"""Analyze the trained models."""

import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # TODO: This is hacky, should be fixed

import tools
from tools import nicename
import task
from model import SingleLayerModel, FullModel

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')

def _easy_save(save_path, str='', dpi=300, pdf=True):
    save_name = save_path.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path, exist_ok=True)
    figname = os.path.join(path, save_name + str)
    plt.savefig(os.path.join(figname + '.png'), dpi=dpi)

    if pdf:
        plt.savefig(os.path.join(figname + '.pdf'), transparent=True)
    plt.close()

def plot_progress(save_path, linestyles=None, alpha = 1, legends= None):
    """Plot progress through training.
        Fixed to allow for multiple plots
    """
    # TODO: This function no longer supports directly plot progress of save_path. Should fix.
    log = tools.load_all_results(save_path, argLast=False)
    def _plot_progress(xkey, ykey):
        figsize = (1.5, 1.2)
        rect = [0.3, 0.3, 0.65, 0.5]
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes(rect)

        ys = log[ykey]
        xs = log[xkey]
        if linestyles is not None:
            for x, y, s in zip(xs, ys, linestyles):
                ax.plot(x, y, alpha= alpha, linestyle=s)
        else:
            for x, y in zip(xs, ys):
                ax.plot(x, y, alpha= alpha)

        if legends is not None:
            # ax.legend(legends, loc=1, bbox_to_anchor=(1.05, 1.2), fontsize=4)
            ax.legend(legends, fontsize=4)

        ax.set_xlabel(nicename(xkey))
        ax.set_ylabel(nicename(ykey))
        if ykey == 'val_acc' and log[ykey].shape[0] == 1:
            plt.title('Final accuracy {:0.3f}'.format(log[ykey][0,-1]), fontsize=7)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks(np.arange(0, log[xkey][0,-1]+2, 10))
        if ykey in ['val_acc', 'glo_score', 'or_glo_score', 'combined_glo_score']:
            ax.set_ylim([0, 1])
            ax.yaxis.set_ticks([0, 0.5, 1.0])
        ax.set_xlim([-1, len(log[xkey][0,:])])

        _easy_save(save_path, '_' + ykey)

    _plot_progress('epoch', 'val_loss')
    _plot_progress('epoch', 'val_acc')
    _plot_progress('epoch', 'glo_score')
    try:
        _plot_progress('epoch', 'or_glo_score')
        _plot_progress('epoch', 'combined_glo_score')
    except:
        pass


def plot_weights(root_path, var_name = 'w_orn', sort_axis = 0, dir_ix = 0):
    """Plot weights.

    Currently this plots OR2ORN, ORN2PN, and OR2PN
    """
    #TODO: fix code
    dirs = [os.path.join(root_path, n) for n in os.listdir(root_path)]
    save_path = dirs[dir_ix]
    config = tools.load_config(save_path)
    # Load network at the end of training
    model_dir = os.path.join(save_path, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict[var_name]

    # if not hasattr(config, 'receptor_layer') or config.receptor_layer == False:
    #     if config.replicate_orn_with_tiling:
    #         w_orn = np.reshape(
    #             w_orn, (config.N_ORN_DUPLICATION, config.N_ORN, config.N_PN))
    #         w_orn = np.swapaxes(w_orn, 0, 1)
    #         w_orn = np.reshape(w_orn, (-1, config.N_PN))

    # Sort for visualization
    if sort_axis == 0:
        ind_max = np.argmax(w_orn, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_orn[:, ind_sort]
    else:
        ind_max = np.argmax(w_orn, axis=1)
        ind_sort = np.argsort(ind_max)
        w_plot = w_orn[ind_sort, :]

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(2.6, 2.6))
    ax = fig.add_axes(rect)
    vlim = np.round(np.max(abs(w_plot)), decimals=1)
    im = ax.imshow(w_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')

    if var_name == 'w_orn':
        plt.title('ORN-PN connectivity after training', fontsize=7)
        ax.set_xlabel('To PNs', labelpad=-5)
        ax.set_ylabel('From ORNs', labelpad=-5)
    elif var_name == 'w_or':
        plt.title('OR-ORN expression array after training', fontsize=7)
        ax.set_xlabel('ORN', labelpad=-5)
        ax.set_ylabel('Unique OR', labelpad=-5)
    elif var_name == 'w_combined':
        plt.title('OR-PN combined connectivity', fontsize=7)
        ax.set_xlabel('PN', labelpad=-5)
        ax.set_ylabel('OR', labelpad=-5)
    else:
        raise ValueError('unknown variable name for weight matrix: {}'.format(var_name))

    plt.axis('tight')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)
    plt.axis('tight')
    _easy_save(root_path, '_' + var_name + '_' + str(dir_ix))


    # Plot distribution of various connections
    # keys = var_dict.keys()
    # keys = ['model/layer1/bias:0', 'model/layer2/bias:0']
    # for key in keys:
    #     fig = plt.figure(figsize=(2, 2))
    #     plt.hist(var_dict[key].flatten())
    #     plt.title(key)


def plot_activity(save_path):
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    
    tf.reset_default_graph()
    
    # Load dataset
    data_dir = rootpath + config.data_dir[1:]  # this is a hack as well
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, data_dir)
    
    if config.model == 'full':
        CurrentModel = FullModel
    elif config.model == 'singlelayer':
        CurrentModel = SingleLayerModel
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
    
        # Validation
        val_loss, val_acc, glo_act, glo_in, glo_in_pre, kc_act = sess.run(
            [model.loss, model.acc, model.glo, model.glo_in, model.glo_in_pre, model.kc],
            {val_x_ph: val_x, val_y_ph: val_y})
        
        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    
    save_name = save_path.split('/')[-1]
    plt.figure()
    plt.hist(glo_act.flatten(), bins=100)
    plt.title('Glo activity distribution')
    plt.savefig(os.path.join(figpath, save_name + '_pn_activity.pdf'), transparent=True)
    plt.show()
    
    plt.figure()
    plt.hist(kc_act.flatten(), bins=100)
    plt.title('KC activity distribution')
    plt.savefig(os.path.join(figpath, save_name + '_kc_activity.pdf'), transparent=True)
    plt.show()


def plot_results(path, x_key, y_key, loop_key=None, yticks = None):
    """Plot results for varying parameters experiments.

    Args:
        path: str, model save path
        x_key: str, key for the x-axis variable
        y_key: str, key for the y-axis variable
        loop_key: str, key for the value to loop around
    """
    log_plot_dict = {'N_KC': [30, 100, 1000, 10000],
                     'kc_loss_alpha': [.1, 1, 10, 100],
                     'kc_loss_beta': [.1, 1, 10, 100],
                     'initial_pn2kc':[.01, .1, 1]}

    plot_dict = {'kc_inputs': [3, 7, 15, 20, 30, 40, 50]}

    res = tools.load_all_results(path)

    # Sort by x_key
    ind_sort = np.argsort(res[x_key])
    for key, val in res.items():
        res[key] = val[ind_sort]

    fig = plt.figure(figsize=(2, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    if loop_key:
        for x in np.unique(res[loop_key]):
            ind = res[loop_key] == x
            x_plot = res[x_key][ind]
            if x_key in log_plot_dict.keys():
                x_plot = np.log(x_plot)
            ax.plot(x_plot, res[y_key][ind], 'o-', label=str(x))
    else:
        x_plot = res[x_key]
        if x_key in log_plot_dict.keys():
            x_plot = np.log(x_plot)
        ax.plot(x_plot, res[y_key], 'o-')

    if x_key == 'kc_inputs':
        ax.plot([7, 7], [0, 1], '--', color = 'gray')

    if x_key in log_plot_dict.keys():
        xticks = np.array(log_plot_dict[x_key])
        ax.set_xticks(np.log(xticks))
    elif x_key in plot_dict.keys():
        xticks = np.array(plot_dict[x_key])
        ax.set_xticks(xticks)
    else:
        xticks = res[x_key]
        ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlabel(nicename(x_key))
    ax.set_ylabel(nicename(y_key))

    if yticks is None:
        ax.set_yticks([0, 0.5, 1.0])
        plt.ylim([0, 1])
    else:
        ax.set_yticks(yticks)
        plt.ylim(yticks[0]- .1 * yticks[-1], 1.1 * yticks[-1])

    if loop_key:
        l = ax.legend(loc=1, bbox_to_anchor=(1.0, 0.5))
        l.set_title(nicename(loop_key))

    figname = '_' + y_key + '_vs_' + x_key
    if loop_key:
        figname += '_vary' + loop_key

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    _easy_save(path, figname)


if __name__ == '__main__':
    pass