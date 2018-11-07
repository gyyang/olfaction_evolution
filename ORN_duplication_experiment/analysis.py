import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
dirs = [os.path.join(os.getcwd(), n) for n in os.listdir(os.getcwd()) if n[:4] == 'file']
percentages = [0, .25, .5]
fig_dir = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(fig_dir):
    os.mkdir(fig_dir)

glo_score, val_acc, val_loss, train_loss, legends = [], [], [], [],[]
for i, d in enumerate(dirs):
    log_name = os.path.join(d, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)
    glo_score.append(log['glo_score'])
    val_acc.append(log['val_acc'])
    val_loss.append(log['val_loss'])
    train_loss.append(log['train_loss'])
    legends.append(percentages[i])

fig, ax = plt.subplots(nrows=2, ncols =2, figsize=(10,10))
fig.suptitle('Training as a function of noise for duplicated ORNs')

cur_ax = ax[0,0]
cur_ax.plot(np.array(glo_score).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Connectivity score')

cur_ax = ax[0,1]
cur_ax.plot(np.array(val_acc).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Validation Accuracy')

cur_ax = ax[1,0]
cur_ax.plot(np.array(train_loss).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Training Loss')

cur_ax = ax[1,1]
cur_ax.plot(np.array(val_loss).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Validation Loss')

plt.savefig(os.path.join(fig_dir, 'summary.png'))

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
fig, ax = plt.subplots(nrows=3, ncols = 2, figsize=(10,10))
for i, cur_w in enumerate(w):
    ind_max = np.argmax(cur_w, axis=0)
    ind_sort = np.argsort(ind_max)
    cur_w_sorted = cur_w[:, ind_sort]

    ax[i,0].imshow(cur_w, cmap='RdBu_r', vmin = -vlim, vmax = vlim)
    helper(ax[i,0])
    plt.title('Noise: ' + str(percentages[i]))
    ax[i,1].imshow(cur_w_sorted, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    helper(ax[i,1])
    plt.title('Sorted: Noise: ' + str(percentages[i]))

plt.savefig(os.path.join(fig_dir, 'weights.png'))

fig = plt.figure()
ind_max = np.argmax(w[0], axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w[0][:, ind_sort]
plt.imshow(w_plot, cmap='RdBu_r', vmin = -vlim, vmax = vlim)
helper(plt.gca())
plt.savefig(os.path.join(fig_dir, 'big_weight.png'))
