"""Analyze the trained models."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import tools


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
glo_score_list = [tools.compute_glo_score(w)[0] for w in w_orns]

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
