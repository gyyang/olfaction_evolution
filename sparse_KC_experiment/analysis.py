import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools

mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd(), 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
fig_dir = os.path.join(os.getcwd(), 'figures')
list_of_legends = [False, True]
list_of_legends = ['KC loss=' + str(n) for n in list_of_legends]
tools.plot_summary(dirs, fig_dir, list_of_legends, 'KC sparse constraint')

w = []
wkc = []
for i, d in enumerate(dirs):
    model_dir = os.path.join(d, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict['w_orn']
        w.append(w_orn)
        wkc.append(var_dict['layer2/kernel'])


fig, ax = plt.subplots(nrows=3, ncols = 2, figsize=(10,10))
for i, cur_wkc in enumerate(wkc):
    sparsity = np.count_nonzero(cur_wkc >0, axis= 1) / cur_wkc.shape[1]
    ax[i,0].hist(sparsity, bins=20, range=(0,1))
    ax[i,0].set_title(list_of_legends[i])


plt.savefig(os.path.join(fig_dir, 'connections.png'))
