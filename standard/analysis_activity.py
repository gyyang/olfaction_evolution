import os

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import standard.analysis as sa
from tools import nicename
import tools
import task
import settings

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'


use_torch = settings.use_torch


def load_activity_tf(save_path, lesion_kwargs=None):
    """Load model activity.

    Returns:

    """
    import tensorflow as tf
    from model import SingleLayerModel, FullModel, NormalizedMLP
    # # Reload the network and analyze activity
    config = tools.load_config(save_path)
    config.label_type = 'sparse'

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

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
        # results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    return {'glo_in': glo_in, 'glo': glo_out,
            'kc_in': kc_in, 'kc': kc_out}


def load_activity_torch(save_path, lesion_kwargs=None):
    import torch
    from torchmodel import FullModel
    # Reload the network and analyze activity
    config = tools.load_config(save_path)

    # Load dataset
    train_x, train_y, val_x, val_y = task.load_data(config.data_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with torch.no_grad():
        model = FullModel(config=config)
        model.load()
        model.to(device)
        model.readout()

        if lesion_kwargs is not None:
            for key, val in lesion_kwargs.items():
                model.lesion_units(key, val)

        # validation
        val_data = torch.from_numpy(val_x).float().to(device)
        val_target = torch.from_numpy(val_y).long().to(device)

        model.eval()
        results = model(val_data, val_target)

    for key, val in results.items():
        try:
            results[key] = val.numpy()
        except AttributeError:
            pass

    return results


def load_activity(save_path, lesion_kwargs=None):
    if use_torch:
        return load_activity_torch(save_path, lesion_kwargs)
    else:
        return load_activity_tf(save_path, lesion_kwargs)


def plot_activity(save_path):
    results = load_activity(save_path)
    save_name = save_path.split('/')[-1]
    plt.figure()
    plt.hist(results['glo'].flatten(), bins=100)
    plt.title('Glo activity distribution')
    tools.save_fig(save_path, save_name + '_pn_activity')

    plt.figure()
    plt.hist(results['kc'].flatten(), bins=100)
    plt.title('KC activity distribution')
    tools.save_fig(save_path, save_name + '_kc_activity')


def image_activity(save_path, arg, sort_columns = True, sort_rows = True):
    def _image(data, zticks, name, xlabel='', ylabel=''):
        rect = [0.2, 0.15, 0.6, 0.65]
        rect_cb = [0.82, 0.15, 0.02, 0.65]

        fig = plt.figure(figsize=(2.6, 2.6))
        ax = fig.add_axes(rect)
        cm = 'Reds'
        im = ax.imshow(data, cmap=cm, vmin=zticks[0], vmax=zticks[1], interpolation='none')
        plt.axis('tight')

        ax.set_ylabel(nicename(ylabel))
        ax.set_xlabel(nicename(xlabel))
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.tick_params('both', length=0)
        ax.set_xticks([0, data.shape[1]])
        ax.set_yticks([0, data.shape[0]])


        ax = fig.add_axes(rect_cb)
        cb = plt.colorbar(im, cax=ax)
        cb.set_ticks(zticks)
        cb.outline.set_linewidth(0.5)
        cb.set_label('Activity', fontsize=7, labelpad=5)
        plt.tick_params(axis='both', which='major', labelsize=7)
        cb.ax.tick_params('both', length=0)
        plt.axis('tight')
        tools.save_fig(save_path, '_' + name, pdf=False)

    dirs = tools.get_allmodeldirs(save_path)
    for i, d in enumerate(dirs):
        results = load_activity(d)
        data = results[arg]
        if arg == 'glo_in':
            xlabel = 'PN Input'
            zticks = [0, 4]
        elif arg == 'glo':
            xlabel = 'PN'
            zticks = [0, 4]
        elif arg == 'kc':
            xlabel = 'KC'
            zticks = [0, 1]
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))

        if sort_columns:
            data = np.sort(data, axis=1)[:,::-1]
        if sort_rows:
            ix = np.argsort(np.sum(data, axis=1))
            data = data[ix,:]
        _image(data, zticks=zticks, name = 'image_' + arg + '_' + str(i), xlabel=xlabel, ylabel='Odors')


def _distribution(data, save_path, name, xlabel, ylabel, xrange=None,
                  density=False):
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes((0.27, 0.25, 0.65, 0.65))
    plt.hist(data, bins=30, range=xrange, density=density, align='left')
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 2))

    # xticks = np.linspace(xrange[0], xrange[1], 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xrange is not None:
        plt.xlim(xrange)
    # ax.set_xticks(xticks)
    plt.locator_params(axis='x', nbins=3)
    plt.locator_params(axis='y', nbins=3)

    # ax.set_yticks(np.linspace(0, yrange, 3))
    # plt.ylim([0, yrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    tools.save_fig(save_path, '_' + name, pdf=True)


def distribution_activity(save_path, var_names=None):
    dirs = tools.get_allmodeldirs(save_path)
    if var_names is None:
        var_names = ['kc', 'glo']
    elif isinstance(var_names, str):
        var_names = [var_names]

    for d in dirs:
        results = load_activity(d)
        for var_name in var_names:
            data = results[var_name].flatten()
            xlabel = tools.nicename(var_name)
            ylabel = 'Distribution'
            name = 'dist_' + var_name + '_' + tools.get_model_name(d)
            figpath = tools.get_experiment_name(d)
            _distribution(data, figpath, name=name, density=True,
                          xlabel=xlabel, ylabel=ylabel)


def sparseness_activity(save_path, var_names, activity_threshold=0.,
                        lesion_kwargs=None, figname=None):
    """Plot the sparseness of activity.

    Args:
        save_path: model path
        arg: str, the activity to plot
    """
    dirs = tools.get_allmodeldirs(save_path)
    if figname is None:
        figname = ''

    if isinstance(var_names, str):
        var_names = [var_names]

    for d in dirs:
        results = load_activity(d, lesion_kwargs)
        for var_name in var_names:
            data = results[var_name]
            xrange = [-0.05, 1.05]
            if var_name == 'glo':
                name = 'PN'
            elif var_name == 'kc':
                name = 'KC'
            else:
                raise ValueError('Unknown var name', var_name)

            figpath = tools.get_experiment_name(d)

            data1 = np.mean(data > activity_threshold, axis=1)
            fname = figname + 'spars_' + var_name + '_' + tools.get_model_name(d)
            _distribution(data1, figpath, name=fname, density=False,
                          xlabel='Fraction of Active '+name+'s',
                          ylabel='Number of Odors', xrange=xrange)

            data2 = np.mean(data > activity_threshold, axis=0)
            fname = figname + 'spars_' + var_name + '2_' + tools.get_model_name(d)
            _distribution(data2, figpath, name=fname, density=False,
                          xlabel='Fraction of Odors',
                          ylabel='Number of '+name+'s', xrange=xrange)


def plot_mean_activity_sparseness(save_path, arg, xkey,
                                  loop_key=None, select_dict=None):
    dirs = tools.get_allmodeldirs(save_path)

    mean_sparseness = []
    for i, d in enumerate(dirs):
        results = load_activity(d)
        data = results[arg]
        activity_threshold = 0
        data = np.count_nonzero(data > activity_threshold, axis=1) / data.shape[1]
        mean_sparseness.append(data.mean())

    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        setattr(config, arg + '_sparse_mean', mean_sparseness[i])
        tools.save_config(config, d)
    sa.plot_results(save_path, xkey= xkey, ykey= arg + '_sparse_mean',
                    ax_args= {'yticks': [0, .2, .4, .6, .8]},
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                    loop_key=loop_key,
                    select_dict=select_dict)




