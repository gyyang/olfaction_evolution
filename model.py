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
        self.w_orn = tf.constant(0.)

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
        var_dict['w_orn'] = sess.run(self.w_orn)
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)


class SingleLayerModel(Model):
    """Single layer model."""

    def __init__(self, x, y, config, is_training=True):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
            config: configuration class
        """
        super(SingleLayerModel, self).__init__(config.save_path)
        self.config = config

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, config)

        if is_training:
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            self.train_op = optimizer.minimize(self.loss)
            self.saver = tf.train.Saver()

            for v in tf.trainable_variables():
                print(v)

    def _build(self, x, y, config):
        self.logits = tf.layers.dense(x, config.N_ORN, name='layer1')
        self.predictions = tf.sigmoid(self.logits)
        xe_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=y,
                                                          logits=self.logits)
        self.loss = tf.reduce_mean(xe_loss)
        self.acc = tf.constant(0.)


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

    def __init__(self, x, y, config, is_training=True):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
            config: configuration class
            is_training: bool
        """
        super(FullModel, self).__init__(config.save_path)
        self.config = config

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, config, is_training)

        if is_training:
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

    def _build(self, x, y, config, is_training):
        N_GLO = config.N_GLO
        N_KC = config.N_KC
        N_CLASS = config.N_CLASS

        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            w1 = tf.get_variable('kernel', shape=(config.N_ORN, N_GLO),
                                 dtype=tf.float32)
            b_orn = tf.get_variable('bias', shape=(N_GLO,), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())

            if config.direct_glo:
                if config.train_direct_glo:
                    if config.tradeoff_direct_random:
                        alpha = tf.get_variable('alpha', shape=(1,),
                                                dtype=tf.float32,
                                                initializer=tf.zeros_initializer())
                        alpha_gate = tf.nn.sigmoid(alpha)
                        w_orn = (1 - alpha_gate) * w1 + alpha_gate * tf.eye(
                            N_GLO)
                    else:
                        alpha = tf.get_variable('alpha', shape=(1,),
                                                dtype=tf.float32,
                                                initializer=tf.ones_initializer())
                        w_orn = w1 + alpha * tf.eye(N_GLO)
                else:
                    # TODO: Make this work when using more than one neuron per ORN
                    w_orn = w1 + tf.eye(N_GLO)
            else:
                w_orn = w1

            glo = tf.nn.relu(tf.matmul(x, w_orn) + b_orn)

        self.w_orn = w_orn

        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            w2 = tf.get_variable('kernel', shape=(N_GLO, N_KC),
                                 dtype=tf.float32)
            b_glo = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
            if config.sparse_pn2kc:
                w_mask = get_sparse_mask(N_GLO, N_KC, 7)
                w_mask = tf.get_variable(
                    'mask', shape=(N_GLO, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_mask),
                    trainable=False)
                w_glo = tf.multiply(w2, w_mask)
            else:
                w_glo = w2
            kc = tf.nn.relu(tf.matmul(glo, w_glo) + b_glo)

        if config.kc_dropout:
            kc = tf.layers.dropout(kc, 0.5, training=is_training)

        logits = tf.layers.dense(kc, N_CLASS, name='layer3',
                                 reuse=tf.AUTO_REUSE)

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
        pred = tf.argmax(logits, axis=-1)
        # pred = tf.cast(pred, tf.int32)
        # acc = tf.reduce_mean(tf.equal(y_target, pred))
        self.acc = tf.metrics.accuracy(labels=y, predictions=pred)


if __name__ == '__main__':
    experiment = 'robert'
    # experiment = 'peter'
    if experiment == 'peter':
        train_x, train_y = task.generate_repeat()
        val_x, val_y = task.generate_repeat()
        CurrentModel = SingleLayerModel

        class modelConfig():
            N_ORN = 30
            lr = .001
            max_epoch = 100
            batch_size = 256
            save_path = './files/peter_tmp'
            save_freq = 10

    elif experiment == 'robert':
        train_x, train_y, val_x, val_y = task.generate_proto()
        CurrentModel = FullModel

        class modelConfig():
            N_ORN = train_x.shape[1]
            N_GLO = 50
            N_KC = 2500
            N_CLASS = 60
            lr = .001
            max_epoch = 10
            batch_size = 256
            save_path = './files/robert_dev'
            save_freq = 1
            sparse_pn2kc = True
            train_pn2kc = True
            # Whether to have direct glomeruli-like connections
            direct_glo = True
            # Whether the coefficient of the direct glomeruli-like connection
            # motif is trainable
            train_direct_glo = True
            # Whether to tradeoff the direct and random connectivity
            tradeoff_direct_random = False
            # Whether to impose all cross area connections are positive
            sign_constraint = False  # TODO: TBF
            # dropout
            kc_dropout = False
    else:
        raise NotImplementedError

    config = modelConfig()
    batch_size = config.batch_size
    n_batch = train_x.shape[0] // batch_size

    train_x_ph = tf.placeholder(train_x.dtype, train_x.shape)
    train_y_ph = tf.placeholder(train_y.dtype, train_y.shape)
    train_iter, next_element = make_input(train_x_ph, train_y_ph, batch_size)
    model = CurrentModel(next_element[0], next_element[1], config=config)

    val_x_ph = tf.placeholder(val_x.dtype, val_x.shape)
    val_y_ph = tf.placeholder(val_y.dtype, val_y.shape)
    val_model = CurrentModel(val_x_ph, val_y_ph, config=config, is_training=False)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iter.initializer, feed_dict={train_x_ph: train_x,
                                                    train_y_ph: train_y})

        loss = 0
        for ep in range(config.max_epoch):
            for b in range(n_batch):
                loss, _ = sess.run([model.loss, model.train_op])

            # Validation
            val_loss, val_acc = sess.run([val_model.loss, val_model.acc],
                                         {val_x_ph: val_x, val_y_ph: val_y})
            print('[*] Epoch {:d}  train_loss={:0.2f}, val_loss={:0.2f}'.format(ep, loss, val_loss))
            print('Validation accuracy', val_acc)

            if ep % config.save_freq ==0:
                # model.save(epoch=ep)
                model.save_pickle(epoch=ep)





