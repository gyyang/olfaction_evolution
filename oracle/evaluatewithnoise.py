"""Evaluate a trained network with different amount of noise."""

import sys
import os
import pickle

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors.kde import KernelDensity
import tensorflow as tf

rootpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(rootpath)  # This is hacky, should be fixed

import task
from model import FullModel
import tools
import standard.analysis_pn2kc_training as analysis_pn2kc_training
from standard.analysis import _easy_save

# foldername = 'multi_head'
foldername = 'tmp_train'

path = os.path.join(rootpath, 'files', foldername)
figpath = os.path.join(rootpath, 'figures', foldername)

# analysis_pn2kc_training.plot_sparsity(path, dynamic_thres=True)

# TODO: clean up these paths
path = os.path.join(path, '0')
config = tools.load_config(path)
config.data_dir = rootpath + config.data_dir[1:]
config.save_path = rootpath + config.save_path[1:]

# Load dataset
train_x, train_y, val_x, val_y = task.load_data(
        config.dataset, config.data_dir)

def evaluate(noise_level):
    tf.reset_default_graph()
    # Build validation model
    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    config.ORN_NOISE_STD = noise_level
    val_model = FullModel(val_x_ph, val_y_ph, config=config, training=False)
    
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        val_model.load()
    
        # Validation
        val_loss, val_acc, val_acc2 = sess.run(
            [val_model.loss, val_model.acc, val_model.acc2],
            {val_x_ph: val_x, val_y_ph: val_y})
    
    return val_loss, val_acc[1]

noise_levels = [0, 0.1, 0.2]
losses = list()
accs = list()
for noise_level in noise_levels:
    val_loss, val_acc = evaluate(noise_level)
    losses.append(val_loss)
    accs.append(val_acc)


from oracle import directreadout

oa = directreadout.OracleAnalysis()
accs_oracle, losses_oracle = oa.get_losses_by_noise(noise_levels)

plt.figure()
plt.plot(noise_levels, accs_oracle, 'o-', color='black')
plt.plot(noise_levels, accs, 'o-', color='red')
plt.ylabel('Acc')

plt.figure()
plt.plot(noise_levels, losses_oracle, 'o-', color='black')
plt.plot(noise_levels, losses, 'o-', color='red')
plt.ylabel('Loss')