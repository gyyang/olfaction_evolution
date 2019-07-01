import tools
import numpy as np
import matplotlib.pyplot as plt
from tools import nicename
import standard.analysis as sa
import os
import matplotlib as mpl
mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

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
        sa._easy_save(save_path, '_' + name, pdf=False)

    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    for i, d in enumerate(dirs):
        glo_in, glo_out, kc_out, results = sa.load_activity(d)

        if arg == 'glo_in':
            data = glo_in
            xlabel = 'PN Input'
            zticks = [0, 4]
        elif arg == 'glo_out':
            data = glo_out
            xlabel = 'PN'
            zticks = [0, 4]
        elif arg == 'kc_out':
            data = kc_out
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

def _distribution(data, save_path, name, xlabel, ylabel, xrange):
    fig = plt.figure(figsize=(1.5, 1.5))
    ax = fig.add_axes((0.27, 0.25, 0.65, 0.65))
    plt.hist(data, bins=30, range=[xrange[0], xrange[1]], density=False, align='left')

    # xticks = np.linspace(xrange[0], xrange[1], 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
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

    sa._easy_save(save_path, '_' + name, pdf=True)

def distribution_activity(save_path, arg):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    for i, d in enumerate(dirs):
        glo_in, glo_out, kc_out, results = sa.load_activity(d)
        if arg == 'glo_in':
            data = glo_in.flatten()
            xlabel = 'PN Input'
            zticks = [-10, 10]
        elif arg == 'glo_out':
            data = glo_out.flatten()
            xlabel = 'PN Activity'
            zticks = [0, 10]
        elif arg == 'kc_out':
            data = kc_out.flatten()
            xlabel = 'Fraction of Active KCs'
            zticks = [0, 2]
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))
        ylabel = 'Number of Cells'
        _distribution(data, save_path, name= 'dist_' + arg + '_' + str(i), xlabel=xlabel, ylabel=ylabel, xrange=zticks)


def sparseness_activity(save_path, arg, activity_threshold=0.,
                        lesion_kwargs=None, figname=None):
    """Plot the sparseness of activity.

    Args:
        path: model path
        arg: str, the activity to plot
    """
    if tools._islikemodeldir(save_path):
        dirs = [save_path]
    else:
        dirs = tools._get_alldirs(save_path, model=True, sort=True)
    if figname is None:
        figname = ''
    for i, d in enumerate(dirs):
        glo_in, glo_out, kc_out, results = sa.load_activity(d, lesion_kwargs)
        if arg == 'glo_out':
            data = glo_out
            name = 'PN'
            zticks = [-0.1, 1]
        elif arg == 'kc_out':
            data = kc_out
            name = 'KC'
            zticks = [-0.1, 1]
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))

        # plt.figure()
        # plt.hist(data.flatten())
        # plt.xlabel('Activity')
        # plt.ylabel('Odors')
        # plt.show()

        data1 = np.mean(data > activity_threshold, axis=1)
        figname_new = figname + 'spars_' + arg + '_' + str(i)
        _distribution(data1, save_path, name=figname_new,
                      xlabel='Fraction of Active '+name+'s',
                      ylabel='Number of Odors', xrange=zticks)

        data2 = np.mean(data > activity_threshold, axis=0)
        figname_new = figname + 'spars_' + arg + '2_' + str(i)
        _distribution(data2, save_path, name=figname_new,
                      xlabel='Fraction of Odors',
                      ylabel='Number of '+name+'s', xrange=zticks)


def plot_mean_activity_sparseness(save_path, arg, x_key,
                                  loop_key=None, select_dict=None):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]

    mean_sparseness = []
    for i, d in enumerate(dirs):
        glo_in, glo_out, kc_out, results = sa.load_activity(d)
        if arg == 'glo_out':
            data = glo_out
        elif arg == 'kc_out':
            data = kc_out
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))
        activity_threshold = 0
        data = np.count_nonzero(data > activity_threshold, axis=1) / data.shape[1]
        mean_sparseness.append(data.mean())

    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        setattr(config, arg + '_sparse_mean', mean_sparseness[i])
        tools.save_config(config, d)
    sa.plot_results(save_path, x_key= x_key, y_key= arg + '_sparse_mean',
                    ax_args= {'yticks': [0, .2, .4, .6, .8]},
                    figsize=(1.5, 1.5), ax_box=(0.27, 0.25, 0.65, 0.65),
                    loop_key=loop_key,
                    select_dict=select_dict)




