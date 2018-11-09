import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools

mpl.rcParams['font.size'] = 7

dir = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/vary_PN/files'
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
fig_dir = os.path.join(os.getcwd())

glo_score, val_acc = [], []
for i, d in enumerate(dirs):
    log_name = os.path.join(d, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)
    glo_score.append(log['glo_score'])
    val_acc.append(log['val_acc'])

parameters =np.array(list(range(10,110,10)) + [150, 200, 250])

mpl.rcParams['font.size'] = 10
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
y = [d[-1] for d in glo_score]
ax.plot(np.log10(parameters), y)
ax.set_xlabel('nPNs')
ax.set_ylabel('Glo Score')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
# ax.xaxis.set_ticks(x[::2], parameters[::2])
plt.sca(ax)
ix = [10, 50, 100, 250]
plt.xticks(np.log10(ix), ix)

plt.savefig(os.path.join(fig_dir, 'vary_PN.pdf'))

dir = 'C:/Users/Peter/PycharmProjects/olfaction_evolution/vary_size_experiment/vary_KC/files'
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
fig_dir = os.path.join(os.getcwd())

glo_score, val_acc = [], []
for i, d in enumerate(dirs):
    log_name = os.path.join(d, 'log.pkl')
    with open(log_name, 'rb') as f:
        log = pickle.load(f)
    glo_score.append(log['glo_score'])
    val_acc.append(log['val_acc'])

parameters =np.array([50, 100, 200, 400, 800, 1200, 2500, 5000, 10000, 20000])

mpl.rcParams['font.size'] = 10
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
y = [d[-1] for d in glo_score]
ax.plot(np.log10(parameters), y)
ax.set_xlabel('nKCs')
ax.set_ylabel('Glo Score')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.yaxis.set_ticks([0, 0.5, 1.0])
ax.xaxis.set_ticks([])
# ax.xaxis.set_ticks(x[::2], parameters[::2])
plt.sca(ax)
ix = [50, 200, 2500, 20000]
str = [50, 200, '2.5K', '20K']
plt.xticks(np.log10(ix), str)

plt.savefig(os.path.join(fig_dir, 'vary_KC.pdf'))
