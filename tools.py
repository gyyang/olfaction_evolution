import os
import json

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle

def save_config(config, save_path):
    # Save config (first convert to dictionary)
    config_dict = {k: getattr(config, k) for k in dir(config) if k[0] != '_'}
    with open(os.path.join(save_path, 'config.json'), 'w') as f:
        json.dump(config_dict, f)


def load_config(save_path):
    with open(os.path.join(save_path, 'config.json'), 'r') as f:
        config_dict = json.load(f)

    class Config():
        pass

    config = Config()
    for key, val in config_dict.items():
        setattr(config, key, val)
    return config


def compute_glo_score(w_orn):
    """Compute the glomeruli score in numpy.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_orn, n_pn). This matrix has to be organized
        in the following way: the n_orn neurons are grouped into n_pn groups
        index 0, ..., n_pn_per_orn - 1 is the 0-th group, and so on

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_pn,), all glomeruli scores
    """
    n_orn, n_pn = w_orn.shape
    w_orn_by_pn = abs(w_orn)

    # this code does **not** work for arbitrary orn / pn sizes
    # n_pn_per_orn = n_orn // n_pn
    # w_orn_by_pn = np.reshape(w_orn, (n_orn, n_pn_per_orn, n_pn))
    # w_orn_by_pn = w_orn_by_pn.mean(axis=1)
    # w_orn_by_pn = abs(w_orn_by_pn)

    glo_scores = list()
    for i in range(n_pn):
        w_tmp = w_orn_by_pn[:, i]  # all projections to the i-th PN neuron
        indsort = np.argsort(w_tmp)[::-1]
        w_max = w_tmp[indsort[0]]
        w_second = w_tmp[indsort[1]]
        glo_score = (w_max - w_second) / (w_max + w_second)
        glo_scores.append(glo_score)

    avg_glo_score = np.mean(glo_scores)
    return avg_glo_score, glo_scores

def plot_summary(dirs, fig_dir, list_of_legends, title):
    mpl.rcParams['font.size'] = 7
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)

    glo_score, val_acc, val_loss, train_loss, legends = [], [], [], [], []
    for i, d in enumerate(dirs):
        log_name = os.path.join(d, 'log.pkl')
        with open(log_name, 'rb') as f:
            log = pickle.load(f)
        glo_score.append(log['glo_score'])
        val_acc.append(log['val_acc'])
        val_loss.append(log['val_loss'])
        train_loss.append(log['train_loss'])
        legends.append(list_of_legends[i])

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig.suptitle(title)

    number = len(legends)
    cmap = plt.get_cmap('jet')
    colors = [cmap(i) for i in np.linspace(0, 1, number)]

    cur_ax = ax[0, 0]
    cur_ax.set_color_cycle(colors)
    cur_ax.plot(np.array(glo_score).transpose())
    cur_ax.legend(legends)
    cur_ax.set_xlabel('Epochs')
    cur_ax.set_ylabel('Connectivity score')

    cur_ax = ax[0, 1]
    cur_ax.set_color_cycle(colors)
    cur_ax.plot(np.array(val_acc).transpose())
    cur_ax.legend(legends)
    cur_ax.set_xlabel('Epochs')
    cur_ax.set_ylabel('Validation Accuracy')

    cur_ax = ax[1, 0]
    cur_ax.set_color_cycle(colors)
    cur_ax.plot(np.array(train_loss).transpose())
    cur_ax.legend(legends)
    cur_ax.set_xlabel('Epochs')
    cur_ax.set_ylabel('Training Loss')

    cur_ax = ax[1, 1]
    cur_ax.set_color_cycle(colors)
    cur_ax.plot(np.array(val_loss).transpose())
    cur_ax.legend(legends)
    cur_ax.set_xlabel('Epochs')
    cur_ax.set_ylabel('Validation Loss')

    plt.savefig(os.path.join(fig_dir, 'summary.pdf'))