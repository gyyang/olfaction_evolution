import numpy as np
import os
import tools
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def plot_weight_change_vs_meta_update_magnitude(path, mat, dir_ix, xlim = 25):
    from tools import save_fig
    def _helper_plot(ax, data, color, label):
        ax.plot(np.arange(len(data)), data, color=color, label=label)
        # ax.tick_params(axis='y', labelcolor=color)

    dirs = [os.path.join(path, n) for n in os.listdir(path)]
    dir = dirs[dir_ix]
    epoch_path = os.path.join(dir, 'epoch')
    lr_ix = 0
    list_of_wglo = tools.load_pickle(epoch_path, 'w_glo')
    list_of_worn = tools.load_pickle(epoch_path, 'w_orn')
    list_of_lr = tools.load_pickle(epoch_path, 'model/lr:0')

    weight_diff = []
    for i in range(len(list_of_wglo)-1):
        change_wglo = list_of_wglo[i+1] - list_of_wglo[i]
        change_wglo = np.mean(np.abs(change_wglo))

        change_worn = list_of_worn[i+1] - list_of_worn[i]
        change_worn = np.mean(np.abs(change_worn))
        print([change_worn, change_wglo])
        weight_diff.append(change_worn + change_wglo)

    relevant_lr = []
    for i in range(0, len(list_of_lr)-1):
        relevant_lr.append(list_of_lr[i][lr_ix])

    fig = plt.figure(figsize=(1.5, 1.2))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
    ax_ = ax.twinx()

    _helper_plot(ax, relevant_lr, 'C0', 'Update Rate')
    _helper_plot(ax_, weight_diff, 'C1', r'$\Delta$ Weight')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Magnitude')

    ax.set_xlim([0, xlim])
    ax.set_yticks([0, .1, .2])
    ax.set_ylim([-0.02, .23])
    ax_.set_yticks([])
    ax.legend(loc=1, bbox_to_anchor=(1, .6), frameon=False, fontsize=5)
    ax_.legend(loc=1, bbox_to_anchor=(1, .4), frameon=False, fontsize=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax_.spines["top"].set_visible(False)
    ax_.spines["right"].set_visible(False)

    # mat = mat.replace('/', '_')
    # mat = mat.replace(':0', '')
    save_fig(path, '_{}_change_vs_lr_{}'.format('weight', dir_ix), pdf=True)