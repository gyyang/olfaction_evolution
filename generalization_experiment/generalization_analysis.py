"""Analyze the trained models."""

import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.size'] = 7

experiment_path = os.path.join(os.getcwd(), 'files', 'generalization_experiment')
dirs = [os.path.join(experiment_path, n) for n in os.listdir(experiment_path)]
percentages = [str(10 * i) for i in range(1,11,3)]
glo_score, val_acc, val_loss, train_loss, legends = [], [], [], [],[]
for i, d in enumerate(dirs):
    # if i == 1:
    log_name = os.path.join(d, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)
    glo_score.append(log['glo_score'])
    val_acc.append(log['val_acc'])
    val_loss.append(log['val_loss'])
    train_loss.append(log['train_loss'])
    legends.append(percentages[i])

fig, ax = plt.subplots(nrows=2, ncols =2, figsize=(10,10))
fig.suptitle('Training as a function of percent of odors that generalize')

cur_ax = ax[0,0]
cur_ax.plot(np.array(glo_score).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Connectivity score')

cur_ax = ax[0,1]
cur_ax.plot(np.array(val_acc).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Validation Accuracy')

cur_ax = ax[1,0]
cur_ax.plot(np.array(train_loss).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Training Loss')

cur_ax = ax[1,1]
cur_ax.plot(np.array(val_loss).transpose())
cur_ax.legend(legends)
cur_ax.set_xlabel('Epochs')
cur_ax.set_ylabel('Validation Loss')


plt.savefig(os.getcwd() + '/figures/' + 'summary.png', transparent=True)
plt.show()
