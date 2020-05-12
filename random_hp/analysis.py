import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools
from tools import nicename

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')


path = os.path.join(rootpath, 'files', 'random_hp_mlp')
res = tools.load_all_results(path)
n_model = len(res['val_acc'])

ykey = 'val_acc'
# ykey = 'val_loss'

indsort = np.argsort(res[ykey])

ind_bio = list(res['NEURONS']).index([50, 2500])  # find the bio model

figsize = (1.5, 1.5)
fig = plt.figure(figsize=figsize)
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
ax.plot(res[ykey][indsort], 'o', ms=1)
ax.plot([0, n_model], [res[ykey][ind_bio]]*2)
ax.set_xlabel('Models')
ax.set_ylabel(nicename(ykey))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
plt.savefig(os.path.join(figpath, 'random_hp.pdf'), transparent=True)
plt.savefig(os.path.join(figpath, 'random_hp.png'), dpi=300)



print(res['NEURONS'][indsort][-5:])



# plt.figure()
# plt.plot(res['glo_score'][indsort])