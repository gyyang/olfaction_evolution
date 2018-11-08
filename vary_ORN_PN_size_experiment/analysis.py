import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools

# mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd(), 'vary_KC')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
dirs = dirs
fig_dir = os.path.join(os.getcwd(), 'figures')
list_of_legends = [50, 100, 200, 400, 800, 1200, 2500, 5000, 10000, 20000]
# list_of_legends =list(range(10,110,10)) + [150, 200, 250]
list_of_legends = ['KC:' + str(n) for n in list_of_legends]
tools.plot_summary(dirs, fig_dir, list_of_legends, 'nORN=PN=50, vary nKC')

w = []
for i, d in enumerate(dirs):
    model_dir = os.path.join(d, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict['w_orn']
        w.append(w_orn)

def helper(ax):
    plt.sca(ax)
    plt.axis('tight', ax= ax)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs')
    ax.set_ylabel('From ORNs')
    # cb = plt.colorbar()
    # cb.outline.set_linewidth(0.5)
    # cb.set_label('Weight', fontsize=7, labelpad=-10)
    plt.tick_params(axis='both', which='major', labelsize=7)

vlim = .5
fig, ax = plt.subplots(nrows=6, ncols = 2, figsize=(10,10))
for i, cur_w in enumerate(w[::2]):
    ind_max = np.argmax(cur_w, axis=0)
    ind_sort = np.argsort(ind_max)
    cur_w_sorted = cur_w[:, ind_sort]

    ax[i,0].imshow(cur_w, cmap='RdBu_r', vmin = -vlim, vmax = vlim)
    helper(ax[i,0])
    plt.title(str(list_of_legends[i*2]))
    ax[i,1].imshow(cur_w_sorted, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    helper(ax[i,1])
    plt.title('Sorted')

plt.savefig(os.path.join(fig_dir, 'weights.pdf'))

fig = plt.figure()
ind_max = np.argmax(w[-1], axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w[-1][:, ind_sort]
plt.imshow(w_plot, cmap='RdBu_r', vmin = -vlim, vmax = vlim)
helper(plt.gca())
plt.savefig(os.path.join(fig_dir, 'big_weight.pdf'))
