"""Analyze the trained models."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt


# save_path = './files/robert_dev'
save_path = './files/robert_bio'
# save_path = './files/peter_tmp'

log_name = os.path.join(save_path, 'log.pkl')
with open(log_name, 'rb') as f:
    log = pickle.load(f)

plt.figure()
plt.plot(log['epoch'], log['val_loss'])
plt.plot(log['epoch'], log['train_loss'])
plt.xlabel('Epochs')
plt.ylabel('Train / Val Loss')

plt.figure()
plt.plot(log['epoch'], log['glo_score'])
plt.xlabel('Epochs')
plt.ylabel('Glo Score')

# Load network at the end of training
model_dir = os.path.join(save_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_orn = var_dict['w_orn']

# Sort for visualization
ind_max = np.argmax(w_orn, axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w_orn[:, ind_sort]

fig = plt.figure(figsize=(10, 10))
ax = plt.axes()
vlim = np.max(abs(w_plot))
plt.imshow(w_plot, cmap= 'RdBu_r', vmin= -vlim, vmax= vlim)
plt.colorbar()
plt.axis('tight')
plt.show()
# ax.yaxis.set_major_locator(ticker.MultipleLocator(input_config.NEURONS_PER_ORN))
# ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# name = './W_ep_' + format(ep, '02d') + '.png'
# path_name = os.path.join(save_path, name)
# fig.savefig(path_name, bbox_inches='tight',dpi=300)
# plt.close(fig)
