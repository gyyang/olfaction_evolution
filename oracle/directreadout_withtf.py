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
# train_y -= 1
# val_y -= 1

# Oracle network
data_x, data_y = val_x, val_y
w_oracle = 2*prototype.T
b_oracle = - np.diag(np.dot(prototype, prototype.T))
w_oracle = np.concatenate((np.zeros((w_oracle.shape[0], 1)), w_oracle), axis=1)
b_oracle = np.array([-100] + list(b_oracle))


tf.reset_default_graph()
x = tf.placeholder(tf.float32, (data_x.shape[0], 50))
y = tf.placeholder(tf.int32, (data_y.shape[0]))
alpha = tf.placeholder(tf.float32, ())

w = tf.get_variable('kernel', shape=(50, 100), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_oracle))
b = tf.get_variable('bias', shape=(100,), dtype=tf.float32,
                    initializer=tf.constant_initializer(b_oracle))

x2 = x + tf.random_normal(x.shape, stddev=0.1)
logits = (tf.matmul(x2, w) + b) * alpha

loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                              logits=logits)
pred = tf.argmax(logits, axis=-1)
acc = tf.metrics.accuracy(labels=y, predictions=pred)

alphas = np.logspace(-1, 1, 50)
losses = list()
accs = list()

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
with tf.Session(config=tf_config) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for a in alphas:
        loss_, acc_, logits_ = sess.run([loss, acc, logits],
                               feed_dict={x: data_x, y: data_y, alpha:a})
        losses.append(loss_)
        accs.append(acc_[1])
    print(np.min(losses), np.max(accs))
    
i = np.argmin(losses)

plt.figure()
plt.plot(alphas, losses, 'o-')
plt.ylabel('Loss')
plt.title('Min loss {:0.2f} at alpha {:0.2f}'.format(losses[i], alphas[i]))
plt.figure()
plt.plot(alphas, accs, 'o-')
plt.ylim([0, 1])
plt.ylabel('Acc')
plt.title('Max: {:0.2f}'.format(np.max(accs)))
# =============================================================================
# plt.figure()
# plt.hist(logits_.flatten())
# =============================================================================
