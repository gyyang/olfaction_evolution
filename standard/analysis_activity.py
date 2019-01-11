import tools
import numpy as np
import matplotlib.pyplot as plt
from tools import nicename
import standard.analysis as sa
import os

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
    fig = plt.figure(figsize=(2.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
    plt.hist(data, bins= 50, range=[xrange[0], xrange[1]], density=False, align='left')

    xticks = np.linspace(xrange[0], xrange[1], 5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xlim(xrange)
    ax.set_xticks(xticks)

    # ax.set_yticks(np.linspace(0, yrange, 3))
    # plt.ylim([0, yrange])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    sa._easy_save(save_path, '_' + name, pdf=False)

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
            xlabel = 'KC Activity'
            zticks = [0, 2]
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))
        ylabel = 'Number of Cells'
        _distribution(data, save_path, name= 'dist_' + arg + '_' + str(i), xlabel=xlabel, ylabel=ylabel, xrange=zticks)

def sparseness_activity(save_path, arg):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    for i, d in enumerate(dirs):
        glo_in, glo_out, kc_out, results = sa.load_activity(d)
        if arg == 'glo_out':
            data = glo_out
            xlabel = 'PN Activity'
            zticks = [0, 1]
        elif arg == 'kc_out':
            data = kc_out
            xlabel = 'KC Activity'
            zticks = [0, 1]
        else:
            raise ValueError('data type not recognized for image plotting: {}'.format(arg))
        ylabel = 'Number of Odors'
        activity_threshold = 0
        data = np.count_nonzero(data > activity_threshold, axis=1) / data.shape[1]
        _distribution(data, save_path, name= 'spars_' + arg + '_' + str(i), xlabel=xlabel, ylabel=ylabel, xrange=zticks)





