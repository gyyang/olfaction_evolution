import numpy as np
import os
import tools
import matplotlib.pyplot as plt

def plot_weight_change_vs_meta_update_magnitude(path, mat):
    from standard.analysis import _easy_save
    def _helper_plot(ax, data, color, label):
        ax.set_xlabel('Epochs')
        ax.set_ylabel(label, color=color)
        ax.plot(np.arange(len(data)), data, color=color)
        # ax.tick_params(axis='y', labelcolor=color)

    epoch_path = os.path.join(path, '0', 'epoch')
    lr_ix = 0
    list_of_wglo = tools.load_pickle(epoch_path, mat)
    list_of_lr = tools.load_pickle(epoch_path, 'model/lr:0')

    weight_diff = []
    for i in range(len(list_of_wglo)-1):
        change = list_of_wglo[i+1] - list_of_wglo[i]
        change = np.sum(np.abs(change))
        weight_diff.append(change)

    relevant_lr = []
    for i in range(0, len(list_of_lr)-1):
        relevant_lr.append(list_of_lr[i][lr_ix])

    fig = plt.figure(figsize=(3.5, 2))
    ax = fig.add_axes([0.2, 0.2, 0.6, 0.7])
    _helper_plot(ax, relevant_lr, 'blue', 'Update Magnitude')
    ax_ = ax.twinx()
    _helper_plot(ax_, weight_diff, 'orange', 'Weight Change Magnitude')
    mat = mat.replace('/', '_')
    mat = mat.replace(':0', '')
    _easy_save(path, '_{}_change_vs_learning_rate'.format(mat), pdf=True)