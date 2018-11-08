import tools
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import task
import numpy as np
from model import FullModel

mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd(), 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
# dirs = dirs[:3]
fig_dir = os.path.join(os.getcwd(), 'figures')
list_of_legends = ['batch_norm', None]
tools.plot_summary(dirs, fig_dir, list_of_legends, 'Scaling')

# # Reload the network and analyze activity
fig, ax = plt.subplots(nrows=4, ncols=2)
for i, d in enumerate(dirs):
    config = tools.load_config(d)

    tf.reset_default_graph()
    CurrentModel = FullModel

    # Build validation model
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, is_training=False)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        # Validation
        val_loss, val_acc, glo_act, glo_in, glo_in_pre = sess.run(
            [model.loss, model.acc, model.glo, model.glo_in, model.glo_in_pre],
            {val_x_ph: val_x, val_y_ph: val_y})

        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

    ax[i,0].hist(glo_act.flatten(), bins=100, range =(0, 5))
    ax[i,0].set_title('Activity: ' + str(list_of_legends[i]))
    sparsity = np.count_nonzero(glo_act >0, axis= 1) / glo_act.shape[1]
    ax[i,1].hist(sparsity, bins=20, range=(0,1))
    ax[i,1].set_title('Sparseness')

ax[3,0].hist(val_x.flatten(), bins=20, range= (0,5))
ax[3,0].set_title('OR activation')

plt.savefig(os.path.join(fig_dir, 'norm.png'))