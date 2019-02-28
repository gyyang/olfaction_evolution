import numpy as np
import os
import standard.analysis as sa
import tools
import matplotlib.pyplot as plt
import task
import tensorflow as tf
from model import FullModel
import matplotlib as mpl

mpl.rcParams['font.size'] = 7
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.family'] = 'arial'

def get_correlation_coefficients(save_path, variable):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    list_of_corr_coef = []

    for i, d in enumerate(dirs):
        if variable == 'glo':
            glo_in, glo_out, kc_out, results = sa.load_activity(d)
            mat = glo_out
        elif variable == 'input':
            config = tools.load_config(d)
            train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)
            mat = val_x
        else:
            glo_in, glo_out, kc_out, results = sa.load_activity(d)
            mat = glo_out

        corrcoef = np.corrcoef(mat, rowvar=False)

        mask = ~np.eye(corrcoef.shape[0], dtype=bool)
        flattened_corrcoef = corrcoef[mask]
        list_of_corr_coef.append(np.mean(flattened_corrcoef))


    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    for i, d in enumerate(dirs):
        config = tools.load_config(d)
        setattr(config, variable + '_activity_corrcoef', list_of_corr_coef[i])
        tools.save_config(config, d)

def correlation_across_epochs(save_path, legend):
    dirs = [os.path.join(save_path, n) for n in os.listdir(save_path)]
    ys = []
    for i, d in enumerate(dirs):
        list_of_corr_coef = []
        dir_with_epoch = os.path.join(d, 'epoch')
        epoch_dirs = [os.path.join(dir_with_epoch, x) for x in os.listdir(dir_with_epoch)]
        for epoch_dir in epoch_dirs:
            glo_in, glo_out, kc_out, results = load_activity_helper(d, epoch_dir)
            mat = glo_out
            corrcoef = np.corrcoef(mat, rowvar=False)
            mask = ~np.eye(corrcoef.shape[0], dtype=bool)
            nanmask = ~np.isnan(corrcoef)
            flattened_corrcoef = corrcoef[np.logical_and(mask, nanmask)]
            list_of_corr_coef.append(np.mean(flattened_corrcoef))
        ys.append(list_of_corr_coef)

    y = ys[0]
    figsize = (1.5, 1.2)
    rect = [0.3, 0.3, 0.65, 0.5]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes(rect)
    yticks = [0, 0.5, 1.0]
    ylim = [-.05, 1]
    xlim = len(y)
    ax.plot(np.transpose(ys))
    xticks = np.arange(0, xlim, 10)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Correlation')
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_ylim(ylim)
    ax.set_xlim([0, len(y) - 1])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.legend(legend, fontsize=4, frameon = False)

    from standard.analysis import _easy_save
    _easy_save(save_path, '_correlation_progress', dpi=500)

def load_activity_helper(config_path, epoch_path):
    '''
    Loads model activity from tensorflow
    :param config_path:
    :return:
    '''

    # # Reload the network and analyze activity
    config = tools.load_config(config_path)
    config.data_dir = config.data_dir #hack
    train_x, train_y, val_x, val_y = task.load_data(config.dataset, config.data_dir)

    tf.reset_default_graph()
    CurrentModel = FullModel

    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    model = CurrentModel(val_x_ph, val_y_ph, config=config, training=False)
    model.save_path = epoch_path

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        model.load()

        # Validation
        glo_out, glo_in, kc_out, logits = sess.run(
            [model.glo, model.glo_in, model.kc, model.logits],
            {val_x_ph: val_x, val_y_ph: val_y})
        results = sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
    return glo_in, glo_out, kc_out, results