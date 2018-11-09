
import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')

skip_orn2pns = [True, False, True]
skip_pn2kcs = [False, False, True]

logs = list()
for i in range(3):
    save_name = 'skip_orn2pn' + str(skip_orn2pns[i]) + 'skip_pn2kc' + str(skip_pn2kcs[i])

    save_path = os.path.join(rootpath, 'files', save_name)

    log_name = os.path.join(save_path, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)
    logs.append(log)

labels = ['Identity ORN-PN', 'Random ORN-PN', 'Linear read-out']

# Validation accuracy
figsize = (1.5, 1.8)
rect = [0.3, 0.2, 0.65, 0.4]
fig = plt.figure(figsize=figsize)
ax = fig.add_axes(rect)
for i in range(3):
    log = logs[i]
    ax.plot(log['epoch'], log['val_acc'], label=labels[i])
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation accuracy')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1]+2, 10))
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.set_ylim([0, 1])
ax.set_xlim([0, len(log['epoch'])])
ax.legend(loc=1, bbox_to_anchor=(1.0, 1.9))
plt.savefig(os.path.join(figpath, 'varybasearchitecture_valacc.pdf'), transparent=True)

