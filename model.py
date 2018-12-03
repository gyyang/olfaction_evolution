"""Model file."""

import os
import pickle

import numpy as np
import tensorflow as tf
from configs import FullConfig, SingleLayerConfig

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
        var_dict['w_glo'] = sess.run(self.w_glo)
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)


class SingleLayerModel(Model):
    """Single layer model."""

    def __init__(self, x, y, config=None, training=True):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN)
            y: tf placeholder or iterator element (batch_size, N_GLO)
            config: configuration class
        """
        if config is None:
            self.config = FullConfig()

        super(SingleLayerModel, self).__init__(self.config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, config)

        if training:
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


def _normalize(inputs, norm_type, training=True):
    """Summarize different forms of normalization."""
    if norm_type is not None:
        if norm_type == 'layer_norm':
            # Apply layer norm before activation function
            outputs = tf.contrib.layers.layer_norm(
                inputs, center=True, scale=True)
        elif norm_type == 'batch_norm':
            # Apply layer norm before activation function
            outputs = tf.layers.batch_normalization(
                inputs, center=True, scale=True, training=training)
        elif norm_type == 'batch_norm_nocenterscale':
            # Apply layer norm before activation function
            outputs = tf.layers.batch_normalization(
                inputs, center=False, scale=False, training=training)
        else:
            raise ValueError('Unknown pn_norm type {:s}'.format(norm_type))
    else:
        outputs = inputs

    return outputs

def _sparse_range(sparse_degree):
    range = 2.0 / sparse_degree
    return range

def _sparse_std(n_in, n_out, sparse_degree):
    fan_in = sparse_degree
    fan_out = (n_out / n_in) * sparse_degree
    variance = 2 / (fan_in + fan_out)
    return np.sqrt(variance)

class FullModel(Model):
    """Full 3-layer model."""

    def __init__(self, x, y, config=None, training=True):
        """Make model.

        Args:
            x: tf placeholder or iterator element (batch_size, N_ORN * N_ORN_DUPLICATION)
            y: tf placeholder or iterator element (batch_size, N_CLASS)
            config: configuration class
            training: bool
        """
        if config is None:
            config = FullConfig
        self.config = config

        super(FullModel, self).__init__(self.config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, training)

        if training:
            optimizer = tf.train.AdamOptimizer(self.config.lr)

            excludes = list()
            if 'train_orn2pn' in dir(self.config) and not self.config.train_orn2pn:
                # TODO: this will also exclude batch norm vars, is that right?
                excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope='model/layer1')
            if not self.config.train_pn2kc:
                # excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                #                               scope='model/layer2')
                excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope= 'model/layer2/kernel:0')
            if not self.config.train_kc_bias:
                excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                              scope= 'model/layer2/bias:0')
            var_list = [v for v in tf.trainable_variables() if v not in excludes]

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, training):
        N_ORN = self.config.N_ORN * self.config.N_ORN_DUPLICATION
        N_PN = self.config.N_PN
        N_KC = self.config.N_KC
        self.loss = 0

        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            if self.config.direct_glo:
                range = _sparse_range(1)
                std = _sparse_std(N_ORN, N_PN, 1)
            else:
                range = _sparse_range(N_ORN)
                std = _sparse_std(N_ORN, N_PN, N_ORN)

            w1 = tf.get_variable('kernel', shape=(N_ORN, N_PN), dtype=tf.float32,
                                 initializer= tf.random_uniform_initializer(0.0, range))
            b_orn = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                    initializer=tf.constant_initializer(0))

            if self.config.direct_glo:
                # alpha = tf.get_variable('alpha', shape=(1,), dtype=tf.float32,
                #                         initializer=tf.constant_initializer(0.5))
                # w_orn = w1 + alpha * tf.eye(N_PN)
                # w_orn = w1 * tf.eye(N_PN)
                mask = np.repeat(np.eye(N_PN), self.config.N_ORN_DUPLICATION, axis=0)
                w_orn = w1 * mask
            else:
                w_orn = w1

            if self.config.sign_constraint_orn2pn:
                w_orn = tf.abs(w_orn)

            glo_in_pre = tf.matmul(x, w_orn) + b_orn
            glo_in = _normalize(glo_in_pre, self.config.pn_norm_pre, training)
            if self.config.skip_orn2pn:
                glo_in = x
            glo = tf.nn.relu(glo_in)
            glo = _normalize(glo, self.config.pn_norm_post, training)

        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            if self.config.skip_orn2pn:
                N_USE = N_ORN
            else:
                N_USE = N_PN

            if self.config.sparse_pn2kc:
                range = _sparse_range(self.config.kc_inputs)
                std = _sparse_std(N_USE, N_KC, self.config.kc_inputs)
            else:
                range = _sparse_range(N_USE)
                std = _sparse_std(N_USE, N_KC, N_USE)

            if self.config.uniform_pn2kc:
                initializer = tf.constant_initializer(range)
            else:
                initializer = tf.random_uniform_initializer(0, range)

            w2 = tf.get_variable('kernel', shape=(N_USE, N_KC), dtype=tf.float32,
                                 initializer= initializer)
            b_glo = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer=tf.constant_initializer(self.config.kc_bias))

            if self.config.sparse_pn2kc:
                w_mask = get_sparse_mask(N_USE, N_KC, self.config.kc_inputs)
                w_mask = tf.get_variable(
                    'mask', shape=(N_USE, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_mask),
                    trainable=False)
                w_glo = tf.multiply(w2, w_mask)
            else:
                w_glo = w2

            if self.config.sign_constraint_pn2kc:
                w_glo = tf.abs(w_glo)

            # KC input before activation function
            kc_in = tf.matmul(glo, w_glo) + b_glo
            kc_in = _normalize(kc_in, self.config.kc_norm_pre, training)
            if 'skip_pn2kc' in dir(self.config) and self.config.skip_pn2kc:
                kc_in = glo
            kc = tf.nn.relu(kc_in)
            kc = _normalize(kc, self.config.kc_norm_post, training)

        if self.config.kc_dropout:
            kc = tf.layers.dropout(kc, 0.5, training=training)

        if self.config.kc_loss:
            self.loss += tf.reduce_mean(kc) * 10

        if self.config.label_type == 'combinatorial':
            n_logits = self.config.N_COMBINATORIAL_CLASS
        else:
            n_logits = self.config.N_CLASS
        logits = tf.layers.dense(kc, n_logits, name='layer3', reuse=tf.AUTO_REUSE)

        if self.config.label_type == 'combinatorial':
            self.loss += tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        elif self.config.label_type == 'one_hot':
            self.loss += tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            self.acc = tf.metrics.accuracy(labels=tf.argmax(y, axis=-1),
                                           predictions=tf.argmax(logits,axis=-1))
        elif self.config.label_type == 'sparse':
            self.loss += tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
            pred = tf.argmax(logits, axis=-1)
            self.acc = tf.metrics.accuracy(labels=y, predictions=pred)
        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")

        self.w_orn = w_orn
        self.w_glo = w_glo
        self.glo_in = glo_in
        self.glo_in_pre = glo_in_pre
        self.glo = glo
        self.kc_in = kc_in
        self.kc = kc
        self.logits = logits
