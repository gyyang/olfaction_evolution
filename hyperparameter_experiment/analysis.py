"""Analyze the trained models."""

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
import task
from model import SingleLayerModel, FullModel

mpl.rcParams['font.size'] = 7

# save_name = 'no_threshold_onehot'
save_name = 'test_fix_orn2pn'
# save_name = 'test'

save_path = os.path.join(rootpath, 'files', save_name)
figpath = os.path.join(rootpath, 'figures')

log_name = os.path.join(save_path, 'log.pkl')
with open(log_name, 'rb') as f:
    log = pickle.load(f)


# Validation loss
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
ax.plot(log['epoch'], np.log10(log['val_loss']))
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation loss')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1], 5))
# ax.set_ylim([0, 1])
# plt.savefig(os.path.join(figpath, save_name + '_valacc.pdf'), transparent=True)


# Validation accuracy
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
ax.plot(log['epoch'], log['val_acc'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Validation accuracy')
plt.title('Final accuracy {:0.3f}'.format(log['val_acc'][-1]))
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1], 5))
ax.set_ylim([0, 1])
plt.savefig(os.path.join(figpath, save_name + '_valacc.pdf'), transparent=True)

# Glo score
fig = plt.figure(figsize=(2, 2))
ax = fig.add_axes([0.25, 0.25, 0.65, 0.65])
ax.plot(log['epoch'], log['glo_score'])
ax.set_xlabel('Epochs')
ax.set_ylabel('Glo score')
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks(np.arange(0, log['epoch'][-1], 5))
ax.set_ylim([0, 1])
# plt.savefig(os.path.join(figpath, save_name+'_gloscore.pdf'), transparent=True)
plt.savefig(os.path.join(figpath, save_name+'_gloscore.png'), transparent=True)


# Load network at the end of training
model_dir = os.path.join(save_path, 'model.pkl')
with open(model_dir, 'rb') as f:
    var_dict = pickle.load(f)
    w_orn = var_dict['w_orn']

# Sort for visualization
ind_max = np.argmax(w_orn, axis=0)
ind_sort = np.argsort(ind_max)
w_plot = w_orn[:, ind_sort]

rect = [0.15, 0.15, 0.7, 0.7]
rect_cb = [0.87, 0.15, 0.02, 0.7]
fig = plt.figure(figsize=(3, 3))
ax = fig.add_axes(rect)
vlim = np.round(np.max(abs(w_plot)), decimals=1)
im = ax.imshow(w_plot, cmap= 'RdBu_r', vmin=-vlim, vmax=vlim,
               interpolation='nearest')
plt.axis('tight')
for loc in ['bottom','top','left','right']:
    ax.spines[loc].set_visible(False)
ax.tick_params('both', length=0)
ax.set_xlabel('To PNs')
ax.set_ylabel('From ORNs')
ax = fig.add_axes(rect_cb)
cb = plt.colorbar(im, cax=ax, ticks=[-vlim, vlim])
cb.outline.set_linewidth(0.5)
cb.set_label('Weight', fontsize=7, labelpad=-10)
plt.tick_params(axis='both', which='major', labelsize=7)
plt.axis('tight')
plt.savefig(os.path.join(figpath, save_name + '_worn.pdf'), transparent=True)
plt.show()

# Plot distribution of various connections
# keys = var_dict.keys()
keys = ['model/layer1/bias:0', 'model/layer2/bias:0']
for key in keys:
    fig = plt.figure(figsize=(2, 2))
    plt.hist(var_dict[key].flatten())
    plt.title(key)


# # Reload the network and analyze activity
config = tools.load_config(save_path)

tf.reset_default_graph()

# Load dataset
data_dir = rootpath + config.data_dir[1:]  # this is a hack as well
train_x, train_y, val_x, val_y = task.load_data(config.dataset, data_dir)

if config.model == 'full':
    CurrentModel = FullModel
elif config.model == 'singlelayer':
    CurrentModel = SingleLayerModel
else:
    raise ValueError('Unknown model type ' + str(config.model))

# Build validation model
val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
model.save_path = rootpath + model.save_path[1:]

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    model.load()

    # Validation
    val_loss, val_acc, glo_act, glo_in, glo_in_pre, kc_act = sess.run(
        [model.loss, model.acc, model.glo, model.glo_in, model.glo_in_pre, model.kc],
        {val_x_ph: val_x, val_y_ph: val_y})
    
    results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))


plt.figure()
plt.hist(glo_act.flatten(), bins=100)
plt.title('Glo activity distribution')
plt.savefig(os.path.join(figpath, save_name + '_pn_activity.pdf'), transparent=True)
plt.show()

plt.figure()
plt.hist(kc_act.flatten(), bins=100)
plt.title('KC activity distribution')
plt.savefig(os.path.join(figpath, save_name + '_kc_activity.pdf'), transparent=True)
plt.show()