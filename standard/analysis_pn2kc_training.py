import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from tools import nicename
import utils

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # TODO: This is hacky, should be fixed
mpl.rcParams['font.size'] = 7
figpath = os.path.join(rootpath, 'figures')
thres = 0.05

def plot_sparsity(dir):
    save_name = dir.split('/')[-1]
    path = os.path.join(figpath, save_name)
    os.makedirs(path,exist_ok=True)
    dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
    titles = ['Before Training', 'After Training']
    yrange = [1, 0.5]

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

    for i, d in enumerate(dirs):
        wglo = utils.load_pickle(os.path.join(d,'epoch'), 'w_glo')
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

    def _plot_distribution(data, savename, title, xrange, yrange):
        fig = plt.figure(figsize=(3, 2))
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

        plt.savefig(savename + '.png', dpi=500)

    for i, d in enumerate(dirs):
        wglo = utils.load_pickle(os.path.join(d,'epoch'), 'w_glo')
        wglo = [wglo[0]] + [wglo[-1]]
        for j, w in enumerate(wglo):
            w[np.isnan(w)] = 0
            distribution = w.flatten()
            save_name = os.path.join(path, 'distribution_' + str(i) + '_' + str(j))
            _plot_distribution(distribution, save_name, title= titles[j], xrange= 1.0, yrange = 5000)


# if __name__ == '__main__':
#     dir = "../files/train_KC_claws"
#     plot_sparsity(dir)
#     plot_distribution(dir)


