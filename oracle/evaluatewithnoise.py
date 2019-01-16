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
from model import FullModel, OracleNet
import tools
from oracle import directreadout

mpl.rcParams['font.size'] = 7


# foldername = 'standard_net'
foldername = 'standard_oracle'

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
    # val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    val_model = OracleNet(val_x_ph, val_y_ph, config=config, training=False)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        # val_model.load()
    
        # Validation
        val_loss, val_acc = sess.run(
            [val_model.loss, val_model.acc],
            {val_x_ph: val_x, val_y_ph: val_y})
    
    return val_loss, val_acc


def evaluate_withnoise():
    noise_levels = np.linspace(0, 0.3, 20)
    losses = list()
    accs = list()
    for noise_level in noise_levels:
        val_loss, val_acc = evaluate('orn_noise_std', noise_level)
        losses.append(val_loss)
        accs.append(val_acc)
    
    oa = directreadout.OracleAnalysis()
    
    alphas = [4]
    # alphas = np.logspace(-1, 1, 4)
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

    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
    ax.plot(noise_levels, losses_oracle[:, 0], '-', color='black', label='oracle')
    # ax.plot(rates, losses, '-', color='red', label='standard')
    ax.plot(noise_levels, losses, '-', color='red', label='oracle_tf')
    plt.xlabel('ORN Drop out rate')
    plt.ylabel('Loss')
    plt.legend()
    # plt.savefig('evaluateloss_withdropout.png', dpi=500)


def evaluate_withdropout():
    rates = np.linspace(0, 0.9, 5)
    losses = list()
    accs = list()
    for rate in rates:
        val_loss, val_acc = evaluate('orn_dropout_rate', rate)
        losses.append(val_loss)
        accs.append(val_acc)
    
    oa = directreadout.OracleAnalysis()
    
    accs_oracle, losses_oracle = oa.get_losses_by_dropout(rates, alpha=4)
    
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
    ax.plot(rates, accs_oracle, '-', color='black', label='oracle')
    # ax.plot(rates, accs, '-', color='red', label='standard')
    ax.plot(rates, accs, '-', color='red', label='oracle_tf')
    plt.xlabel('ORN Drop out rate')
    plt.ylabel('Acc')
    plt.legend()
    plt.savefig('evaluateacc_withdropout.png', dpi=500)
    
    fig = plt.figure(figsize=(2,2))
    ax = fig.add_axes([0.3, 0.3, 0.6, 0.6])
    ax.plot(rates, losses_oracle, '-', color='black', label='oracle')
    # ax.plot(rates, losses, '-', color='red', label='standard')
    ax.plot(rates, losses, '-', color='red', label='oracle_tf')
    plt.xlabel('ORN Drop out rate')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('evaluateloss_withdropout.png', dpi=500)
    

if __name__ == '__main__':
    # evaluate_withnoise()
    evaluate_withdropout()