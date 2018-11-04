"""Model file."""

import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import tensorflow as tf
import pickle as pkl

import input


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


class Model(object):
    """Network model."""

    def __init__(self, x, y):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        config = modelConfig()
        lr = config.lr

        input_config = input.smallConfig()
        y_dim = input_config.N_ORN

        self.logits = tf.layers.dense(x, y_dim)
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = self.logits)
        self.loss = tf.reduce_mean(xe_loss)
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess):
        loss = 0
        for ep in range(n_epoch):
            for b in range(n_batch):
                # cur_inputs, cur_labels = sess.run(next_element)
                # feed_dict = {self.x: cur_inputs, self.y: cur_labels}
                # output, loss, _ = sess.run([self.predictions, self.loss, self.train_op], feed_dict)
                output, loss, _ = sess.run(
                    [self.predictions, self.loss, self.train_op])

            if (ep %2 == 0 and ep!= 0):
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

    def test(self, sess, x, y):
        raise NotImplementedError
        feed_dict = {self.x: x, self.y: y}
        output, loss = sess.run([self.predictions, self.loss], feed_dict)
        print('[!] Test_loss=%.2f' % (loss))
        return loss


if __name__ == '__main__':
    in_x, in_y = input.generate()
    x_test, y_test = input.generate()

    input_config = input.smallConfig()

    config = modelConfig()
    save_path = config.save_path
    n_epoch = config.epoch
    batch_size = config.batch_size
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    n_batch = in_x.shape[0] // batch_size
    features = in_x
    labels = in_y
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    train_iter, next_element = make_input(features_placeholder, labels_placeholder, batch_size)
    model = Model(next_element[0], next_element[1])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer,
                 feed_dict={features_placeholder: features,
                            labels_placeholder: labels})

        model.train(sess)


