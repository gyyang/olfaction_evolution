import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import utils
from tools import nicename

def pretty_plot_last_epoch(data, xlabel, ylabel, fig_dir, parameters, ylim, xticks, log=True):
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    if log:
        params = np.log10(parameters)
    else:
        params = parameters

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])

    y = [x[-1] for x in data]
    ax.plot(params, y, marker='o', markersize= 2)
    ax.plot([7, 7], [-1, 10], '--', color='gray')
    ax.set_ylim(0, ylim[-1])
    ax.set_yticks(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(False)
    ax.set_xticks(xticks)
    # plt.xticks(params[::skip], parameters[::skip])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, ylabel + '.png'), dpi=300)


param = "kc_inputs"
condition = "vary_KC_claws/uniform_bias_trainable"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]
titles = ['GloScore', 'Accuracy', 'Validation Loss', 'Training Loss']
ylims = [[0, .5, 1], [0, .5, 1], [0, 3], [0, 3]]
xticks = [1, 7, 15, 30, 50]
for d, t, yl in zip(data, titles, ylims):
    pretty_plot_last_epoch(d, xlabel = nicename(param), ylabel = t, fig_dir = fig_dir, parameters = parameters,
                           xticks= xticks, ylim= yl, log=False)

param = "N_KC"
condition = "vary_N_KC_trainable_with_loss"

mpl.rcParams['font.size'] = 7
fig_dir = os.path.join(os.getcwd(), condition, 'figures')
dir = os.path.join(os.getcwd(), condition, 'files')

configs, glo_score, val_acc, val_loss, train_loss = utils.load_results(dir)
parameters = [getattr(config, param) for config in configs]
list_of_legends = [param +': ' + str(n) for n in parameters]
data = [glo_score, val_acc, val_loss, train_loss]
titles = ['GloScore', 'Accuracy', 'Validation Loss', 'Training Loss']
ylims = [[0, .5, 1], [0, .5, 1], [0, 3], [0, 3]]
xticks = [50, 2500, 10000]
for d, t, yl in zip(data, titles, ylims):
    pretty_plot_last_epoch(d, xlabel=nicename(param), ylabel=t, fig_dir=fig_dir, parameters=parameters,
                           xticks = xticks, ylim=yl, log=False)