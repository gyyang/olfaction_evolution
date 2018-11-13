import os
import numpy as np
import pickle
import tools
import matplotlib.pyplot as plt

def _load_results(root_dir, save_name):
    dir = os.path.join(root_dir, save_name, 'files')
    dirs = [os.path.join(root_dir, dir, n) for n in os.listdir(dir)]

    glo_score = []
    val_acc = []
    config = []
    n_pns = np.zeros(len(dirs))
    n_kcs = np.zeros(len(dirs))
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        config.append(tools.load_config(d))
        glo_score.append(log['glo_score'])
        val_acc.append(log['val_acc'])
    return glo_score, val_acc, config

def adjust(ax):
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_ticks([0, 0.5, 1.0])
    ax.xaxis.set_ticks([])
    ax.set_ylim([0, 1])

def plot_weights(w_orn, name, arg_sort):

    # Sort for visualization
    if arg_sort:
        ind_max = np.argmax(w_orn, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_orn[:, ind_sort]
        str = ''
    else:
        w_plot = w_orn
        str = 'unsorted_'

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect)
    vlim = .5
    im = ax.imshow(w_plot, cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    plt.title('ORN-PN connectivity after training')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs', labelpad=-5)
    ax.set_ylabel('From ORNs', labelpad=-5)
    ax.set_xticks([0, w_plot.shape[1]])
    ax.set_yticks([0, w_plot.shape[0]])
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    plt.savefig(os.path.join(name + str + '_worn.png'),
                dpi=400,
                transparent=True)

def plot_small_weights(w_orn, name, arg_sort):

    # Sort for visualization
    if arg_sort:
        ind_max = np.argmax(w_orn, axis=0)
        ind_sort = np.argsort(ind_max)
        w_plot = w_orn[:, ind_sort]
        str = ''
    else:
        w_plot = w_orn
        str = 'unsorted_'

    rect = [0.15, 0.15, 0.65, 0.65]
    rect_cb = [0.82, 0.15, 0.02, 0.65]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect)
    vlim = .5
    im = ax.imshow(w_plot[:50,:10], cmap='RdBu_r', vmin=-vlim, vmax=vlim,
                   interpolation='none')
    plt.axis('tight')
    plt.title('ORN-PN connectivity after training')
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs', labelpad=-5)
    ax.set_ylabel('From ORNs', labelpad=-5)
    ax.set_xticks([0, 5, 10])
    ax.set_yticks(range(0,51,10))
    ax = fig.add_axes(rect_cb)
    cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
    cb.outline.set_linewidth(0.5)
    cb.set_label('Weight', labelpad=-10)
    plt.tick_params(axis='both', which='major')
    plt.axis('tight')
    plt.savefig(os.path.join(name + str + '_worn.png'),
                dpi=400,
                transparent=True)