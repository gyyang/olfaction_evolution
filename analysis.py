"""Analyze the trained models."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 7

save_name = 'robert_bio'

save_path = './files/' + save_name

log_name = os.path.join(save_path, 'log.pkl')
with open(log_name, 'rb') as f:
    log = pickle.load(f)

fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.plot(log['epoch'], log['val_acc'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation accuracy')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks([0, 5, 10])
ax.set_ylim([0, 1])
plt.savefig('figures/' + save_name + '_valacc.pdf', transparent=True)

fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.plot(log['epoch'], log['glo_score'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Glo score')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks([0, 5, 10])
ax.set_ylim([0, 1])
plt.savefig('figures/' + save_name + '_gloscore.pdf', transparent=True)


# Load network at the end of training
model_dir = os.path.join(save_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_orn = var_dict['w_orn']

# Sort for visualization
ind_max = np.argmax(w_orn, axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w_orn[:, ind_sort]

rect = [0.15, 0.15, 0.7, 0.7]
rect_cb = [0.87, 0.15, 0.02, 0.7]
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes(rect)
vlim = np.round(np.max(abs(w_plot)), decimals=1)
im = ax.imshow(w_plot, cmap= 'RdBu_r', vmin=-vlim, vmax=vlim,
               interpolation='nearest')
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)
ax.tick_params('both', length=0)
ax.set_xlabel('To PNs')
ax.set_ylabel('From ORNs')
ax = fig.add_axes(rect_cb)
cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
cb.outline.set_linewidth(0.5)
cb.set_label('Weight', fontsize=7, labelpad=-10)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.axis('tight')
plt.savefig('figures/' + save_name + '_worn.pdf', transparent=True)
plt.show()
