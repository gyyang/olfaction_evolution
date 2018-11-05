"""Model file."""

import os
import pickle

import numpy as np
import tensorflow as tf

import task


def make_input(x, y, batch_size):
    data = tf.data.Dataset.from_tensor_slices((x, y))
    data = data.shuffle(int(1E6)).batch(tf.cast(batch_size, tf.int64)).repeat()
    train_iter = data.make_initializable_iterator()
    next_element = train_iter.get_next()
    return train_iter, next_element


class Model(object):
    """Abstract Model class."""

    def __init__(self, save_path):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path
        self.saver = None

    def save(self, epoch=None):
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, str(epoch))
        save_path = os.path.join(save_path, 'model.ckpt')
        save_path = self.saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, 'model.pkl')

        sess = tf.get_default_session()
        var_dict = {v.name: sess.run(v) for v in tf.trainable_variables()}
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)


class SingleLayerModel(Model):
    """Single layer model."""

    def __init__(self, x, y, config):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
            config: configuration class
        """
        super(SingleLayerModel, self).__init__(config.save_path)
        self.config = config

        input_config = task.smallConfig()
        y_dim = input_config.N_ORN

        self.logits = tf.layers.dense(x, y_dim, name='layer1')
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = self.logits)
        self.loss = tf.reduce_mean(xe_loss)
        self.acc = tf.constant(0.)

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()

        for v in tf.trainable_variables():
            print(v)


def get_sparse_mask(nx, ny, non):
    """Generate a binary mask.

    The mask will be of size (nx, ny)
    For all the nx connections to each 1 of the ny units, only non connections are 1.

    Args:
        nx: int
        ny: int
        non: int, must not be larger than nx

    Return:
        mask: numpy array (nx, ny)
    """
    mask = np.zeros((nx, ny))
    mask[:non] = 1
    for i in range(ny):
        np.random.shuffle(mask[:, i])  # shuffling in-place
    return mask.astype(np.float32)


class FullModel(Model):
    """Full 3-layer model."""

    def __init__(self, x, y, config):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
            config: configuration class
        """
        super(FullModel, self).__init__(config.save_path)
        self.config = config

        N_GLO = config.N_GLO
        N_KC = config.N_KC
        N_CLASS = config.N_CLASS

        glo = tf.layers.dense(x, N_GLO, activation=tf.nn.relu, name='layer1')
        weights = tf.get_default_graph().get_tensor_by_name(
            os.path.split(glo.name)[0] + '/kernel:0')

        if config.sparse_pn2kc:
            with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
                w = tf.get_variable('kernel', shape=(N_GLO, N_KC), dtype=tf.float32)
                w_mask = get_sparse_mask(N_GLO, N_KC, 7)
                w_mask = tf.constant(w_mask, dtype=tf.float32)
                b = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                kc = tf.nn.relu(tf.matmul(glo, tf.multiply(w, w_mask)) + b)
        else:
            kc = tf.layers.dense(glo, N_KC, activation=tf.nn.relu, name='layer2')
        logits = tf.layers.dense(kc, N_CLASS, name='layer3')

        xe_loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
        # weight_loss = tf.reduce_mean(tf.square(weights))
        # activity_loss = tf.reduce_mean(glo)
        self.loss = xe_loss
        pred = tf.argmax(logits, axis=-1)
        # pred = tf.cast(pred, tf.int32)
        # acc = tf.reduce_mean(tf.equal(y_target, pred))
        self.acc = tf.metrics.accuracy(labels=y, predictions=pred)

        optimizer = tf.train.AdamOptimizer(self.config.lr)

        if config.train_pn2kc:
            var_list = tf.trainable_variables()
        else:
            excludes = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layer2')
            var_list = [v for v in tf.trainable_variables() if v not in excludes]

        self.train_op = optimizer.minimize(self.loss, var_list=var_list)
        self.saver = tf.train.Saver()

        print('Training variables')
        for v in var_list:
            print(v)


if __name__ == '__main__':
    experiment = 'robert'
    if experiment == 'peter':
        features, labels = task.generate_repeat()
        CurrentModel = SingleLayerModel

        class modelConfig():
            lr = .001
            max_epoch = 100
            batch_size = 256
            save_path = './files/peter_tmp'
            save_freq = 10

    elif experiment == 'robert':
        features, labels = task.generate_proto()
        CurrentModel = FullModel

        class modelConfig():
            N_GLO = 50
            N_KC = 2500
            N_CLASS = 1000
            lr = .001
            max_epoch = 10
            batch_size = 256
            save_path = './files/robert_tmp'
            save_freq = 1
            sparse_pn2kc = True
            train_pn2kc = False
    else:
        raise NotImplementedError

    config = modelConfig()
    batch_size = config.batch_size
    n_batch = features.shape[0] // batch_size

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)
    train_iter, next_element = make_input(features_placeholder, labels_placeholder, batch_size)

    model = CurrentModel(next_element[0], next_element[1], config=config)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer,
                 feed_dict={features_placeholder: features,
                            labels_placeholder: labels})

        loss = 0
        for ep in range(config.max_epoch):
            for b in range(n_batch):
                loss, _ = sess.run([model.loss, model.train_op])

            # TODO: do validation here
            print('[*] Epoch %d  total_loss=%.2f' % (ep, loss))

            if ep % config.save_freq ==0:
                # model.save(epoch=ep)
                model.save_pickle(epoch=ep)





