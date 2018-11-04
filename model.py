"""Model file."""

import os
import pickle

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
    """Model."""

    def __init__(self, x, y, save_path=None):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        super(SingleLayerModel, self).__init__(save_path)
        if save_path is None:
            save_path = os.getcwd()
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        self.save_path = save_path

        self.config = modelConfig()

        input_config = task.smallConfig()
        y_dim = input_config.N_ORN

        self.logits = tf.layers.dense(x, y_dim, name='orn')
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = self.logits)
        self.loss = tf.reduce_mean(xe_loss)

        optimizer = tf.train.AdamOptimizer(self.config.lr)
        self.train_op = optimizer.minimize(self.loss)
        self.saver = tf.train.Saver()


class FullModel(Model):
    """Model."""

    def __init__(self, x, y, save_path=None):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
        """
        super(FullModel, self).__init__(save_path)
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
        self.saver = tf.train.Saver()


if __name__ == '__main__':
    experiment = 'robert'
    if experiment == 'peter':
        features, labels = task.generate_repeat()
        CurrentModel = SingleLayerModel
        save_freq = 10
        max_epoch = 100
        save_path = './files/peter_tmp'
    elif experiment == 'robert':
        features, labels = task.generate_proto()
        CurrentModel = FullModel
        save_freq = 1
        max_epoch = 10
        save_path = './files/robert_tmp'
    else:
        raise NotImplementedError

    config = modelConfig()
    batch_size = config.batch_size
    n_batch = features.shape[0] // batch_size

    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    train_iter, next_element = make_input(features_placeholder, labels_placeholder, batch_size)

    model = CurrentModel(next_element[0], next_element[1], save_path=save_path)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer,
                 feed_dict={features_placeholder: features,
                            labels_placeholder: labels})

        loss = 0
        for ep in range(max_epoch):
            for b in range(n_batch):
                loss, _ = sess.run([model.loss, model.train_op])

            # TODO: do validation here
            print('[*] Epoch %d  total_loss=%.2f' % (ep, loss))

            if ep % save_freq ==0:
                # model.save(epoch=ep)
                model.save_pickle(epoch=ep)





