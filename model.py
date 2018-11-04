import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import input
import pickle as pkl
import seaborn as sns
import os

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

class model():
    '''
    Goal of task is to ask whether the same type of ORN neurons can be made to connect to PNs to de-noise ORN inputs
    '''
    def __init__(self, x, y):
        config = modelConfig()
        lr = config.lr

        input_config = input.smallConfig()
        y_dim = input_config.N_ORN

        self.x = x
        self.y = y
        # self.w = tf.get_variable('W', shape=[x_dim,y_dim], dtype=tf.float32)
        # b = tf.get_variable('B', shape=[y_dim], dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        self.logits = tf.layers.dense(self.x, y_dim)
        # self.logits = tf.matmul(self.x, self.w) + b
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = self.logits)
        self.loss = tf.reduce_mean(xe_loss)
        optimizer = tf.train.AdamOptimizer(lr)
        self.train_op = optimizer.minimize(self.loss)

    def train(self, sess):
        # input_config = input.smallConfig()
        #
        # config = modelConfig()
        # save_path = config.save_path
        # n_epoch = config.epoch
        # batch_size = config.batch_size
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        #
        # n_batch = x.shape[0]//batch_size
        # train_iter, next_element= make_input(x, y, batch_size)
        # sess.run(train_iter.initializer, {self.x: x, self.y: y})

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
        feed_dict = {self.x: x, self.y: y}
        output, loss = sess.run([self.predictions, self.loss], feed_dict)
        print('[!] Test_loss=%.2f' % (loss))
        return loss

def see():
    #TODO
    with open('.W.pkl','rb') as f:
        mat = pkl.load(f)



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
    network = model(next_element[0], next_element[1])


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(train_iter.initializer,
                 feed_dict={features_placeholder: features,
                            labels_placeholder: labels})

        network.train(sess)


