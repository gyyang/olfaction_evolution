"""Analyze the trained models."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt



def compute_glo_score(w_orn):
    """Compute the glomeruli score.

    This function returns the glomeruli score, a number between 0 and 1 that
    measures how close the connectivity is to glomeruli connectivity.

    For one glomeruli neuron, first we compute the average connections from
    each ORN group. Then we sort the absolute connection weights by ORNs.
    The glomeruli score is simply:
        (Max weight - Second max weight) / (Max weight + Second max weight)

    Args:
        w_orn: numpy array (n_neuron, n_orn). This matrix has to be organized
        in the following way: the n_neuron neurons are grouped into n_orn groups
        index 0, ..., n_neuron_per_orn - 1 is the 0-th group, and so on

    Return:
        avg_glo_score: scalar, average glomeruli score
        glo_scores: numpy array (n_orn,), all glomeruli scores
    """
    n_neuron, n_orn = w_orn.shape
    n_neuron_per_orn = n_neuron // n_orn
    w_orn_by_orn = np.reshape(w_orn, (n_orn, n_neuron_per_orn, n_orn))
    w_orn_by_orn = w_orn_by_orn.mean(axis=1)
    w_orn_by_orn = abs(w_orn_by_orn)

    glo_scores = list()
    for i in range(n_orn):
        w_tmp = w_orn_by_orn[:, i]  # all projections to the i-th PN neuron
        indsort = np.argsort(w_tmp)[::-1]
        w_max = w_tmp[indsort[0]]
        w_second = w_tmp[indsort[1]]
        glo_score = (w_max - w_second) / (w_max + w_second)
        glo_scores.append(glo_score)

    avg_glo_score = np.mean(glo_scores)
    return avg_glo_score, glo_scores


def load_w_orn(model_dir):
    model_dir = os.path.join(save_path, model_dir, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict['w_orn']
    return w_orn

# save_path = './files/robert_dev'
save_path = './files/robert_bio'
# save_path = './files/peter_tmp'

model_dirs = os.listdir(save_path)  # should be the epoch name
epochs = np.sort([int(m) for m in model_dirs])
model_dirs = [str(m) for m in epochs]

w_orns = [load_w_orn(m) for m in model_dirs]
glo_score_list = [compute_glo_score(w)[0] for w in w_orns]

plt.figure()
plt.plot(epochs, glo_score_list)
plt.xlabel('Epochs')
plt.ylabel('Glo Score')

# Sort for visualization
w_orn = w_orns[-1]
ind_max = np.argmax(w_orn, axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w_orn[:, ind_sort]

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
vlim = np.max(abs(w_plot))
plt.imshow(w_plot, cmap= 'RdBu_r', vmin= -vlim, vmax= vlim)
plt.colorbar()
plt.axis('tight')
# ax.yaxis.set_major_locator(ticker.MultipleLocator(input_config.NEURONS_PER_ORN))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# name = './W_ep_' + format(ep, '02d') + '.png'
# path_name = os.path.join(save_path, name)
# fig.savefig(path_name, bbox_inches='tight',dpi=300)
# plt.close(fig)
