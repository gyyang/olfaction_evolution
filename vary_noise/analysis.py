import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools

mpl.rcParams['font.size'] = 7

new_rootpath = os.path.join(rootpath, 'files', 'vary_noise2')
fig_dir = os.path.join(rootpath, 'figures')


res = tools.load_all_results(new_rootpath)


ind_sort = np.argsort(res['N_KC'])
for key, val in res.items():
    res[key] = val[ind_sort]

x_key = 'N_KC'
loop_key = 'ORN_NOISE_STD'

for y_key in ['glo_score', 'val_acc']:
    plt.figure()
    for x in np.unique(res[loop_key]):
        ind = res[loop_key] == x
        plt.plot(res[x_key][ind], res[y_key][ind], 'o-', label=str(x))
    plt.legend()