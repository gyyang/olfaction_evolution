import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools
from tools import nicename

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')


path = os.path.join(rootpath, 'files', 'random_hp_mlp')
res = tools.load_all_results(path)

indsort = np.argsort(res['val_acc'])



plt.figure()
plt.plot(res['val_acc'][indsort])

# plt.figure()
# plt.plot(res['glo_score'][indsort])