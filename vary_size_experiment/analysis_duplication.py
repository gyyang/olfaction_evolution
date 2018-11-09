import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools

# mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd(), 'vary_PN_dupORN10x_noNoise')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
dirs = dirs
fig_dir = os.path.join(os.getcwd(), 'figures')
parameters = [1, 5, 10, 25, 50, 100, 200, 500]
parameters = [1, 5, 10, 25, 50]
list_of_legends = ['nPN per ORN:' + str(n) for n in parameters]
glo_score, val_acc, val_loss, train_loss = \
    tools.plot_summary(dirs, fig_dir, list_of_legends,
                       'nORN = 50, nORN dup = 10, vary nPN per ORN')

titles = ['glomerular score', 'validation accuracy', 'validation loss', 'training loss']
data = [glo_score, val_acc, val_loss, train_loss]
mpl.rcParams['font.size'] = 10
rc = (2,2)
fig, ax = plt.subplots(nrows=rc[0], ncols=rc[1], figsize=(10,10))
fig.suptitle('nORN = 50, nORN dup = 10, vary nPN per ORN')
for i, (d, t) in enumerate(zip(data, titles)):
    ax_i = np.unravel_index(i, dims=rc)
    cur_ax = ax[ax_i]
    x = parameters
    y = [x[-1] for x in d]
    cur_ax.semilogx(parameters, y,  marker='o')
    cur_ax.set_xlabel('nPN dup')
    cur_ax.set_ylabel(t)
    cur_ax.grid(True)
    cur_ax.spines["right"].set_visible(False)
    cur_ax.spines["top"].set_visible(False)
    cur_ax.xaxis.set_ticks_position('bottom')
    cur_ax.yaxis.set_ticks_position('left')
plt.savefig(os.path.join(fig_dir, 'summary_last_epoch.pdf'))

w = []
for i, d in enumerate(dirs):
    model_dir = os.path.join(d, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict['w_orn']
        w.append(w_orn)

#calculate sparseness

rc= (5,2)
fig, ax = plt.subplots(nrows=rc[0], ncols= rc[1], figsize=(10,10))
fig.suptitle('')
for i, cw in enumerate(w):
    ax_i = np.unravel_index(i, dims=rc)
    cur_ax = ax[ax_i]
    maxval = np.max(cw, axis=0)
    cur_ax.hist(maxval, bins=50, range=[0, 2])
    cur_ax.set_xlabel('max weight')
    cur_ax.set_ylabel('count')
    cur_ax.set_title('nPN per ORN: ' +str(parameters[i]))
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'sparseness.pdf'))

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
for i, cur_w in enumerate(w):
    ind_max = np.argmax(cur_w, axis=0)
    ind_sort = np.argsort(ind_max)
    cur_w_sorted = cur_w[:, ind_sort]

    ax[i,0].imshow(cur_w, cmap='RdBu_r', vmin = -vlim, vmax = vlim)
    helper(ax[i,0])
    plt.title(str(list_of_legends[i]))
    ax[i,1].imshow(cur_w_sorted, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    helper(ax[i,1])
    plt.title('Sorted')
plt.savefig(os.path.join(fig_dir, 'weights.pdf'))
