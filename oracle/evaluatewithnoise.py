"""Evaluate a trained network with different amount of noise."""

import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import task
from model import FullModel
import tools
from oracle import directreadout

foldername = 'standard_net'

path = os.path.join(rootpath, 'files', foldername)
figpath = os.path.join(rootpath, 'figures', foldername)

# TODO: clean up these paths
path = os.path.join(path, '0')
config = tools.load_config(path)
config.data_dir = rootpath + config.data_dir[1:]
config.save_path = rootpath + config.save_path[1:]

# Load dataset
train_x, train_y, val_x, val_y = task.load_data(
        config.dataset, config.data_dir)

def evaluate(name, value):
    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    if name == 'orn_noise_std':
        config.ORN_NOISE_STD = value
    elif name == 'orn_dropout_rate':
        config.orn_dropout = True
        config.orn_dropout_rate = value
    else:
        raise ValueError('Unknown name', str(name))
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_model.load()
    
        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})
    
    return val_loss, val_acc[1]


def evaluate_withnoise():
    noise_levels = np.linspace(0, 0.3, 20)
    losses = list()
    accs = list()
    for noise_level in noise_levels:
        val_loss, val_acc = evaluate('orn_noise_std', noise_level)
        losses.append(val_loss)
        accs.append(val_acc)
    
    oa = directreadout.OracleAnalysis()
    
    alphas = np.logspace(-1, 1, 4)
    accs_oracle, losses_oracle = oa.get_losses_by_noisealpha(noise_levels, alphas)
    
    plt.figure()
    plt.plot(noise_levels, accs_oracle[:, 0], 'o-', color='black')
    plt.plot(noise_levels, accs, 'o-', color='red')
    plt.xlabel('Noise level')
    plt.ylabel('Acc')
    
    # =============================================================================
    # for i in range(len(alphas)):
    #     plt.figure()
    #     plt.plot(noise_levels, losses_oracle[:, i], 'o-', color='black')
    #     plt.plot(noise_levels, losses, 'o-', color='red')
    #     plt.xlabel('Noise level')
    #     plt.ylabel('Loss')
    # 
    # =============================================================================


if __name__ == '__main__':
    evaluate_withnoise()