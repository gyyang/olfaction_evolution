"""Model file."""

import os
import pickle

import numpy as np
import tensorflow as tf


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
        sess = tf.get_default_session()
        save_path = self.saver.save(sess, save_path)
        print("Model saved in path: %s" % save_path)

    def load(self):
        save_path = self.save_path
        save_path = os.path.join(save_path, 'model.ckpt')
        sess = tf.get_default_session()
        self.saver.restore(sess, save_path)
        print("Model restored from path: {:s}".format(save_path))

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

            for v in tf.trainable_variables():
                print(v)

        self.saver = tf.train.Saver()

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

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, config, is_training):
        N_GLO = config.N_GLO
        N_KC = config.N_KC

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
                                                initializer=tf.constant_initializer(0.5))
                        w_orn = w1 + alpha * tf.eye(N_GLO)
                else:
                    # TODO: Make this work when using more than one neuron per ORN
                    w_orn = w1 + tf.eye(N_GLO)
            else:
                w_orn = w1

            if config.sign_constraint:
                w_orn = tf.abs(w_orn)

            glo = tf.nn.relu(tf.matmul(x, w_orn) + b_orn)

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

            if config.sign_constraint:
                w_glo = tf.abs(w_glo)

            # KC input before activation function
            kc_in = tf.matmul(glo, w_glo) + b_glo


            if 'kc_layernorm' in dir(config) and config.kc_layernorm:
                # Apply layer norm before activation function
                kc_in = tf.contrib.layers.layer_norm(kc_in)

            kc = tf.nn.relu(kc_in)

        if config.kc_dropout:
            kc = tf.layers.dropout(kc, 0.5, training=is_training)

        if config.label_type == 'combinatorial':
            n_logits = config.N_COMBINATORIAL_CLASS
        else:
            n_logits = config.N_CLASS
        logits = tf.layers.dense(kc, n_logits, name='layer3', reuse=tf.AUTO_REUSE)

        if config.label_type == 'combinatorial':
            self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        elif config.label_type == 'one_hot':
            self.loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            self.acc = tf.metrics.accuracy(labels=tf.argmax(y, axis=-1),
                                           predictions=tf.argmax(logits,axis=-1))
        elif config.label_type == 'sparse':
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
            pred = tf.argmax(logits, axis=-1)
            self.acc = tf.metrics.accuracy(labels=y, predictions=pred)
        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")

        self.w_orn = w_orn
        self.glo = glo
        self.kc = kc
        self.logits = logits
