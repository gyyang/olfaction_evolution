"""Model file."""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf

import task


N_GLO = 50
N_KC = 2500
N_CLASS = 1000

class modelConfig():
    lr = .001
    epoch = 100
    batch_size = 100
    save_path = './files'


def make_input(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element


class SingleLayerModel(object):
    """Model."""

    def __init__(self, x, y):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        self.config = modelConfig()

        input_config = task.smallConfig()
        y_dim = input_config.N_ORN

        self.logits = tf.layers.dense(x, y_dim)
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = self.logits)
        self.loss = tf.reduce_mean(xe_loss)

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)


class FullModel(object):
    """Model."""

    def __init__(self, x, y):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        self.config = modelConfig()

        glo = tf.layers.dense(x, N_GLO, activation=tf.nn.relu)
        kc = tf.layers.dense(glo, N_KC, activation=tf.nn.relu)
        logits = tf.layers.dense(kc, N_CLASS)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
        pred = tf.argmax(logits, axis=-1)
        # pred = tf.cast(pred, tf.int32)
        # acc = tf.reduce_mean(tf.equal(y_target, pred))
        self.acc = tf.metrics.accuracy(labels=y, predictions=pred)

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)


if __name__ == '__main__':
    experiment = 'peter'
    if experiment == 'peter':
        features, labels = task.generate_repeat()
        Model = SingleLayerModel
    elif experiment == 'robert':
        features, labels = task.generate_proto()
        Model = FullModel
    else:
        raise NotImplementedError

    config = modelConfig()
    save_path = config.save_path
    batch_size = config.batch_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_batch = features.shape[0] // batch_size

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    train_iter, next_element = make_input(features_placeholder, labels_placeholder, batch_size)

    model = Model(next_element[0], next_element[1])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer,
                 feed_dict={features_placeholder: features,
                            labels_placeholder: labels})

        loss = 0
        for ep in range(config.epoch):
            for b in range(n_batch):
                # cur_inputs, cur_labels = sess.run(next_element)
                # feed_dict = {self.x: cur_inputs, self.y: cur_labels}
                # output, loss, _ = sess.run([self.predictions, self.loss, self.train_op], feed_dict)
                loss, _ = sess.run([model.loss, model.train_op])

            if (ep % 2 == 0 and ep != 0):
                print('[*] Epoch %d  total_loss=%.2f' % (ep, loss))

            # if (ep%10 ==0 and ep != 0):
            #     w = sess.run(self.w)
            #     path_name = os.path.join(save_path, 'W.pkl')
            #     with open(path_name,'wb') as f:
            #         pkl.dump(w, f)
            #     fig = plt.figure(figsize=(10, 10))
            #     ax = plt.axes()
            #     plt.imshow(w, cmap= 'RdBu_r', vmin= -.5, vmax= .5)
            #     plt.colorbar()
            #     plt.axis('tight')
            #     ax.yaxis.set_major_locator(ticker.MultipleLocator(input_config.NEURONS_PER_ORN))
            #     ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            #     name = './W_ep_' + format(ep, '02d') + '.png'
            #     path_name = os.path.join(save_path, name)
            #     fig.savefig(path_name, bbox_inches='tight',dpi=300)
            #     plt.close(fig)



