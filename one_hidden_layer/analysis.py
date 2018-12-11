import sys
import os

import matplotlib.pyplot as plt
import matplotlib as mpl
import task
import train
from model import NormalizedMLP
import tensorflow as tf
import numpy as np
import one_hidden_layer.experiment as train_config

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root_path)  # This is hacky, should be fixed


mpl.rcParams['font.size'] = 7

# new_rootpath = os.path.join(rootpath, 'files', train_config.save_path)
fig_dir = os.path.join(root_path, 'figures')

# Rebuild model to
CurrentModel = NormalizedMLP
config = train_config.config
# hacky way to run it on CPU
config.save_path = '.'+ config.save_path
train_x, train_y, val_x, val_y = task.load_data(config.dataset,
                                                # hacky way to run it on CPU
                                                '.'+config.data_dir)
# Build train model
train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
train_iter, next_element = train.make_input(train_x_ph, train_y_ph, config.batch_size)
model = CurrentModel(next_element[0], next_element[1], config=config)

tf_config = tf.ConfigProto()

with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                train_y_ph: train_y})
    model.load()
    results = sess.run([v for v in tf.trainable_variables() if v.name=='model/layer1/kernel:0'])
    orn_to_kc_w = results[0]
    print(orn_to_kc_w.shape)
    U, S, V = np.linalg.svd(orn_to_kc_w)
    plt.plot(S)
    plt.show()
