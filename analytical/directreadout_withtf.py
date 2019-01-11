import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


path = '../datasets/proto/test_data'

names = ['train_x', 'train_y', 'val_x', 'val_y', 'prototype']
results = [np.load(os.path.join(path, name + '.npy')) for name in names]
train_x, train_y, val_x, val_y, prototype = results

# the 0 class is not used
assert np.min(train_y) == 1
train_y -= 1
val_y -= 1

# Oracle network
data_x, data_y = val_x, val_y
w_oracle = 2*prototype.T
b_oracle = - np.diag(np.dot(prototype, prototype.T))


tf.reset_default_graph()
x = tf.placeholder(tf.float32, (None, 50))
y = tf.placeholder(tf.int32, (None))
alpha = tf.placeholder(tf.float32, ())

w = tf.get_variable('kernel', shape=(50, 99), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_oracle))
b = tf.get_variable('bias', shape=(99,), dtype=tf.float32,
                    initializer=tf.constant_initializer(b_oracle))


logits = (tf.matmul(x, w) + b) * alpha

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                              logits=logits)
pred = tf.argmax(logits, axis=-1)
acc = tf.metrics.accuracy(labels=y, predictions=pred)


tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    loss_, acc_ = sess.run([loss, acc], feed_dict={x: data_x, y: data_y, alpha:20})

    print(loss_, acc_)