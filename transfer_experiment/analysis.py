
import sys
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)

import tools

mpl.rcParams['font.size'] = 7

figpath = os.path.join(rootpath, 'figures')

save_name1 = 'transfer_batchnorm'
save_name2 = 'transfer_layernorm'

save_path1 = os.path.join(rootpath, 'files', save_name1)
save_path2 = os.path.join(rootpath, 'files', save_name2)

logs = list()
for save_path in [save_path1, save_path2]:
    log = tools.load_log(save_path)
    logs.append(log)

config = tools.load_config(save_path)
task_epoch = config.max_epoch  # number of epochs each task is trained

# Validation accuracy
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
colors = [np.array([77, 174, 74])/255., np.array([55,126,184])/255.]
labels = ['BatchNorm', 'LayerNorm']
for i, log in enumerate(logs):
    ax.plot(log['epoch'][-task_epoch:], log['val_acc'][-task_epoch:],
            color=colors[i], label=labels[i])
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation accuracy')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1], 5))
ax.set_ylim([0, 1])
ax.legend(loc=4)
plt.savefig(os.path.join(figpath, 'transferlearning_valacc.pdf'), transparent=True)
