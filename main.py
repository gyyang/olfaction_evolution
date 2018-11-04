import numpy as np
import tensorflow as tf

import task

tf.reset_default_graph()
batch_size = 256
N_ORN = task.N_ORN
N_GLO = 50
N_KC = 2500

x = tf.placeholder(tf.float32, (None, N_ORN))
# =============================================================================
# glo = tf.layers.dense(x, N_GLO, activation=tf.nn.relu)
# kc = tf.layers.dense(glo, N_KC, activation=tf.nn.relu)
# y = tf.layers.dense(kc, task.N_CLASS)
# 
# =============================================================================
y = tf.layers.dense(x, task.N_CLASS)

y_target = tf.placeholder(tf.int32, (None))

loss = tf.losses.sparse_softmax_cross_entropy(labels=y_target, logits=y)
pred = tf.argmax(y, axis=-1)
# pred = tf.cast(pred, tf.int32)
# acc = tf.reduce_mean(tf.equal(y_target, pred))
acc = tf.metrics.accuracy(labels=y_target, predictions=pred)
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    for step in range(100000):
        ind = np.random.randint(0, task.N_TRAIN, size=(batch_size,))
        x_val = task.train_odors[ind, :]
        y_target_val = task.train_labels[ind]

        # x_val = np.random.randn(batch_size, N_ORN).astype(np.float32)
        # y_target_val = np.argmax(x_val, axis=1)

        loss_val, acc_val, y_val, _ = sess.run([loss, acc, y, train_step],
                                      feed_dict={x: x_val, y_target: y_target_val})

        if step % 1000 == 0:
            x_val = task.val_odors
            y_target_val = task.val_labels
            loss_val, acc_val = sess.run([loss, acc],
                                         feed_dict={x: x_val,
                                                    y_target: y_target_val})
            print(acc_val)
            print(loss_val)