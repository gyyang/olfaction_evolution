import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tools
import tensorflow as tf
from model import SingleLayerModel, FullModel
import task

# mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd())
dirs = [os.path.join(dir, n) for n in os.listdir(dir) if n[:1] == '1']
dirs = dirs
fig_dir = os.path.join(dir, 'figures')
list_of_legends = [0,-1,-2,-3,-4]
glo_score, val_acc, val_loss, train_loss = \
    tools.plot_summary(dirs, fig_dir, list_of_legends,
                       'nORN = 50, nORN dup = 10, vary nPN per ORN')

worn, born, wpn, bpn = [], [], [], []
for i, d in enumerate(dirs):
    model_dir = os.path.join(d, 'model.pkl')
    with open(model_dir, 'rb') as f:
        var_dict = pickle.load(f)
        w_orn = var_dict['w_orn']
        worn.append(w_orn)
        born.append(var_dict['model/layer1/bias:0'])
        wpn.append(var_dict['model/layer2/kernel:0'])
        bpn.append(var_dict['model/layer2/bias:0'])
        print(tools.compute_glo_score(w_orn)[0])

def helper(ax):
    plt.sca(ax)
    plt.axis('tight', ax= ax)
    for loc in ['bottom', 'top', 'left', 'right']:
        ax.spines[loc].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xlabel('To PNs')
    ax.set_ylabel('From ORNs')
    plt.tick_params(axis='both', which='major', labelsize=7)

vlim = 2
fig, ax = plt.subplots(nrows=5, ncols = 2, figsize=(10,10))
for i, cur_w in enumerate(worn):
    ind_max = np.argmax(cur_w, axis=0)
    ind_sort = np.argsort(ind_max)
    cur_w_sorted = cur_w[:, ind_sort]


    ax[i,0].imshow(cur_w, cmap='RdBu_r', vmin = -vlim, vmax=vlim)
    helper(ax[i,0])
    plt.title(str(list_of_legends[i]))
    ax[i,1].imshow(cur_w_sorted, cmap='RdBu_r', vmin=-vlim, vmax=vlim)
    helper(ax[i,1])
    plt.title('Sorted')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'weights.png'))


# # Reload the network and analyze activity
fig, ax = plt.subplots(nrows=6, ncols=2)
for i, d in enumerate(dirs):
    config = tools.load_config(d)

    tf.reset_default_graph()
    CurrentModel = FullModel

    # Build validation model
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        # Validation
        val_loss, val_acc, glo_act, glo_in, glo_in_pre, kc = sess.run(
            [model.loss, model.acc, model.glo, model.glo_in, model.glo_in_pre, model.kc],
            {val_x_ph: val_x, val_y_ph: val_y})

        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ax[i,0].hist(kc.flatten(), bins=100, range =(0, 5))
    ax[i,0].set_title('Activity: ' + str(list_of_legends[i]))
    sparsity = np.count_nonzero(kc >0, axis= 1) / kc.shape[1]
    ax[i,1].hist(sparsity, bins=20, range=(0,1))
    ax[i,1].set_title('Sparseness')

plt.savefig(os.path.join(fig_dir, 'activity.png'))