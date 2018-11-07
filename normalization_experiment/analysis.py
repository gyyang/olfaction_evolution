import tools
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import task
import numpy as np
from model import SingleLayerModel, FullModel

mpl.rcParams['font.size'] = 7
dir = os.path.join(os.getcwd(), 'files')
dirs = [os.path.join(dir, n) for n in os.listdir(dir)]
fig_dir = os.path.join(os.getcwd(), 'figures')
list_of_legends = ['batch norm', 'sparse norm', 'no norm']
tools.plot_summary(dirs, fig_dir, list_of_legends, 'ORN Duplication')

# # Reload the network and analyze activity
for i, d in enumerate(dirs):
    config = tools.load_config(d)

    tf.reset_default_graph()
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)
    if config.model == 'full':
        CurrentModel = FullModel
    elif config.model == 'singlelayer':
        CurrentModel = SingleLayerModel
    else:
        raise ValueError('Unknown model type ' + str(config.model))

    # Build validation model
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

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0,0].hist(glo_act.flatten(), bins=100)
    ax[0,0].set_title('Glo activity distribution')
    sparsity = np.count_nonzero(glo_act, axis= 1) / glo_act.shape[1]
    ax[0,1].hist(sparsity, bins=100)
    ax[0,1].set_title('Activity Sparseness distribution')
    plt.savefig(os.path.join(fig_dir, str(i) + '_pn_activity.png'))