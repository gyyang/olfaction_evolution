
import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import tools

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')

save_name1 = 'transfer_batchnorm'
save_name2 = 'transfer_layernorm'

save_path1 = os.path.join(rootpath, 'files', save_name1)
save_path2 = os.path.join(rootpath, 'files', save_name2)

logs = list()
for save_path in [save_path1, save_path2]:
    with open(os.path.join(save_path, 'log.pkl'), 'rb') as f:
        log = pickle.load(f)
    logs.append(log)

config = tools.load_config(save_path)
task_epoch = config.max_epoch  # number of epochs each task is trained

# Validation accuracy
fig = plt.figure(figsize=(4, 2))
ax = fig.add_axes([0.1, 0.25, 0.3, 0.65])
for log in logs:
    ax.plot(log['epoch'][task_epoch:], log['val_acc'][task_epoch:])
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation accuracy')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1], 5))
ax.set_ylim([0, 1])
plt.savefig(os.path.join(figpath, 'transferlearning_valacc.pdf'), transparent=True)
