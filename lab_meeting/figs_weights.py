import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
from matplotlib import pylab
import utils
from utils import load_results, adjust
from utils import plot_weights

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

mpl.rcParams['font.size'] = 12

plt.style.use('dark_background')
fig_dir = 'C:/Users/Peter/Dropbox/TALKS/LAB MEETINGS/2018.11.29'
fig_dir = os.path.join(fig_dir)
root_dir = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/duplication'
save_name = ['.25_noise','one_epoch','no_noise', 'no_training']
for s in save_name:
    dir = os.path.join(root_dir,s, 'files')
    file_folders = [os.path.join(dir, n) for n in os.listdir(dir)]
    w = []


    for i, d in enumerate(file_folders):
        model_dir = os.path.join(d, 'model.pkl')
        with open(model_dir, 'rb') as f:
            var_dict = pickle.load(f)
            w_orn = var_dict['w_orn']
            w.append(w_orn)
            name = os.path.join(fig_dir, s)
            # plot_weights(w_orn, name, 1)

file_path = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/vary_PN/files/09'
model_dir = os.path.join(file_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_orn = var_dict['w_orn']
    w.append(w_orn)
    name = os.path.join(fig_dir, 'vary_pn_weight')
    plot_weights(w_orn, name, 1)
