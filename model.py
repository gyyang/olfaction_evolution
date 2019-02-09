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
        """Save model using pickle."""
        pass

    def lesion_units(self, name, units, verbose=False):
        """Lesion units given by units.

        Args:
            name: name of the layer to lesion
            units : can be None, an integer index, or a list of integer indices
        """
        sess = tf.get_default_session()
        # Convert to numpy array
        if units is None:
            return
        elif not hasattr(units, '__iter__'):
            units = np.array([units])
        else:
            units = np.array(units)

        # This lesioning will work for both RNN and GRU
        v = [tmp for tmp in tf.trainable_variables() if tmp.name == name][0]
        # Connection weights
        v_val = sess.run(v)
        v_val[units, :] = 0
        sess.run(v.assign(v_val))

        if verbose:
            print('Lesioned units:')
            print(units)


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


def get_sparse_mask(nx, ny, non, complex=False, nOR=50):
    """Generate a binary mask.

    The mask will be of size (nx, ny)
    For all the nx connections to each 1 of the ny units, only non connections are 1.

    If complex == True, KCs cannot receive the connections from the same OR from duplicated ORN inputs.
    Assumed to be 'repeat' style duplication.

    Args:
        nx: int
        ny: int
        non: int, must not be larger than nx

    Return:
        mask: numpy array (nx, ny)
    """
    mask = np.zeros((nx, ny))

    if not complex:
        mask[:non] = 1
        for i in range(ny):
            np.random.shuffle(mask[:, i])  # shuffling in-place
    else:
        OR_ixs = [np.arange(i, nx, nOR) for i in range(nOR)] # only works for repeat style duplication
        for i in range(ny):
            ix = [np.random.choice(bag) for bag in OR_ixs]
            ix = np.random.choice(ix, non, replace=False)
            mask[ix,i] = 1
    return mask.astype(np.float32)

import normalization
def _normalize(inputs, norm_type, training=True):
    """Summarize different forms of normalization."""
    if norm_type is not None:
        if norm_type == 'layer_norm':
            # Apply layer norm before activation function
            # outputs = tf.contrib.layers.layer_norm(
            #     inputs, center=True, scale=True)
            outputs = tf.contrib.layers.layer_norm(
                inputs, center=True, scale=False)
        elif norm_type == 'batch_norm':
            # Apply layer norm before activation function
            outputs = tf.layers.batch_normalization(
                inputs, center=True, scale=True, training=training)
            # The keras version is not working properly because it's doesn't
            # respect the reuse variable in scope
        elif norm_type == 'batch_norm_nocenterscale':
            # Apply layer norm before activation function
            outputs = tf.layers.batch_normalization(
                inputs, center=False, scale=False, training=training)
        elif norm_type == 'custom':
            outputs = normalization.custom_norm(inputs, center=False, scale=True)
        elif norm_type == 'biology':
            exp = 1
            r_max = tf.get_variable('r_max', shape=(1,), dtype=tf.float32, initializer=tf.constant_initializer(5))
            rho = tf.get_variable('rho', shape=(1,), dtype=tf.float32, initializer=tf.constant_initializer(1))
            m = tf.get_variable('m', shape=(1,), dtype=tf.float32, initializer=tf.constant_initializer(0.01))
            sums = tf.reduce_sum(inputs, axis=1, keepdims=True)
            num = r_max * tf.pow(inputs, exp)
            den = tf.pow(inputs, exp) + rho + tf.pow(m * sums, exp)
            outputs =  tf.divide(num, den)
        elif norm_type == 'activity':
            r_max = tf.get_variable('r_max', shape=(1, 50), dtype=tf.float32, initializer=tf.constant_initializer(40))
            sums = tf.reduce_sum(inputs, axis=1, keepdims=True) + 1e-6
            outputs = r_max * tf.divide(inputs, sums)
        else:
            print('Unknown pn_norm type {:s}. Outputs = Inputs'.format(norm_type))
            outputs = inputs
    else:
        outputs = inputs

    return outputs


def _sparse_range(sparse_degree):
    """Generate range of random variables given connectivity degree."""
    range = 2.0 / sparse_degree
    return range


def _glorot_std(n_in, n_out, sparse_degree):
    fan_in = sparse_degree
    fan_out = (n_out / n_in) * sparse_degree
    variance = 2 / (fan_in + fan_out)
    return np.sqrt(variance)


def _initializer(range, arg):
    """Specify initializer given range and type."""
    if arg == 'constant':
        initializer = tf.constant_initializer(range)
    elif arg == 'uniform':
        initializer = tf.random_uniform_initializer(0, range * 2)
    elif arg == 'normal':
        initializer = tf.random_normal_initializer(0, range)
    elif arg == 'learned':
        initializer = tf.random_normal_initializer(range, .1)
    else:
        initializer = None
    return initializer


def _noise(x, arg, std):
    """Add noise to input."""
    if arg == 'additive':
        x += tf.random_normal(x.shape, stddev=std)
    elif arg == 'multiplicative':
        x += x * tf.random_normal(x.shape, stddev=std)
    elif arg == None:
        pass
    else:
        raise ValueError('Unknown noise model {:s}'.format(arg))
    return x

def _get_oracle(prototype_repr):
    """Given prototype representation, return oracle weights."""
    w_oracle = 2 * prototype_repr.T
    b_oracle = -np.diag(np.dot(prototype_repr, prototype_repr.T))
    return w_oracle, b_oracle


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
                # self.train_op = optimizer.minimize(self.loss, var_list=var_list)

                gvs = optimizer.compute_gradients(self.loss, var_list=var_list)
                self.gradient_norm = [tf.norm(g) for g, v in gvs if g is not None]
                self.var_names = [v.name for g, v in gvs if g is not None]
                self.train_op = optimizer.apply_gradients(gvs)


            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()
        # self.saver = tf.train.Saver(tf.trainable_variables())

    def _build_obsolete(self, x, y, training):
        config = self.config
        N_PN = config.N_PN
        N_KC = config.N_KC
        self.loss = 0

        if config.receptor_layer:
            # Define another layer, layer0, that connects receptors to ORNs
            with tf.variable_scope('layer0', reuse= tf.AUTO_REUSE):
                N_OR = config.N_ORN
                ORN_DUP = config.N_ORN_DUPLICATION
                N_ORN = config.N_ORN * ORN_DUP
                range = 1/N_OR
                initializer = _initializer(range, config.initializer_or2orn)
                w_or = tf.get_variable('kernel', shape=(N_OR, N_ORN), dtype=tf.float32,
                                     initializer=initializer)
                if config.sign_constraint_or2orn:
                    w_or = tf.abs(w_or)

                if config.or2orn_normalization:
                    sums = tf.reduce_sum(w_or, axis=0)
                    w_or = tf.divide(w_or, sums)

                if config.or_bias:
                    b_or = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                            initializer=tf.constant_initializer(-0.01))
                else:
                    b_or = 0

                orn = tf.matmul(x, w_or) + b_or
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
        else:
            if config.replicate_orn_with_tiling:
                # Replicating ORNs through tiling
                ORN_DUP = config.N_ORN_DUPLICATION
                N_ORN = config.N_ORN * ORN_DUP

                assert x.shape[-1] == config.N_ORN
                orn = tf.tile(x, [1, ORN_DUP])
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
            else:
                ORN_DUP = 1
                N_ORN = config.N_ORN
                orn = x
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)

        if config.orn_dropout:
            # This is interpreted as noise, so it's always on
            orn = tf.layers.dropout(orn, config.orn_dropout_rate,
                                    training=True)

        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            if config.sign_constraint_orn2pn:
                initializer = _initializer(_sparse_range(N_ORN), config.initializer_orn2pn)
                bias_initializer = tf.constant_initializer(0)
            else:
                initializer = tf.glorot_normal_initializer
                bias_initializer = tf.glorot_normal_initializer

            w1 = tf.get_variable('kernel', shape=(N_ORN, N_PN),
                                 dtype=tf.float32,
                                 initializer=initializer)

            b_orn = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                    initializer= bias_initializer)

            if config.direct_glo:
                w_orn = np.tile(np.eye(N_PN), (ORN_DUP,1)) / ORN_DUP
                w_orn = tf.constant(value=w_orn, dtype=tf.float32)
                b_orn = np.zeros(N_PN)
            else:
                w_orn = w1

            if config.sign_constraint_orn2pn:
                w_orn = tf.abs(w_orn)

            if config.orn2pn_normalization:
                sums = tf.reduce_sum(w_orn, axis=0, keepdims=True)
                w_orn = tf.divide(w_orn, sums)

            glo_in_pre = tf.matmul(orn, w_orn) + b_orn
            if config.skip_orn2pn:
                glo_in = orn
            else:
                glo_in = _normalize(glo_in_pre, config.pn_norm_pre, training)

            glo = tf.nn.relu(glo_in)
            glo = _normalize(glo, config.pn_norm_post, training)

        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            if config.skip_orn2pn:
                N_USE = N_ORN
            else:
                N_USE = N_PN

            if config.sign_constraint_pn2kc:
                if config.initial_pn2kc == 0:
                    if config.sparse_pn2kc:
                        range = _sparse_range(config.kc_inputs)
                    else:
                        range = _sparse_range(N_USE)
                else:
                    range = config.initial_pn2kc
                initializer = _initializer(range, config.initializer_pn2kc)
                bias_initializer = tf.constant_initializer(config.kc_bias)
            else:
                initializer = tf.glorot_normal_initializer
                bias_initializer = tf.glorot_normal_initializer

            w2 = tf.get_variable('kernel', shape=(N_USE, N_KC), dtype=tf.float32,
                                 initializer= initializer)
            b_glo = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer= bias_initializer)

            if config.sparse_pn2kc:
                if config.skip_orn2pn:
                    w_mask = get_sparse_mask(N_USE, N_KC, config.kc_inputs, complex=True)
                else:
                    w_mask = get_sparse_mask(N_USE, N_KC, config.kc_inputs)
                w_mask = tf.get_variable(
                    'mask', shape=(N_USE, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_mask),
                    trainable=False)
                w_glo = tf.multiply(w2, w_mask)
            else:
                w_glo = w2

            if config.sign_constraint_pn2kc:
                w_glo = tf.abs(w_glo)

            if config.mean_subtract_pn2kc:
                w_glo -= tf.reduce_mean(w_glo, axis=0)

            # KC input before activation function
            kc_in = tf.matmul(glo, w_glo) + b_glo
            kc_in = _normalize(kc_in, config.kc_norm_pre, training)
            if 'skip_pn2kc' in dir(self.config) and config.skip_pn2kc:
                kc_in = glo
            kc = tf.nn.relu(kc_in)
            kc = _normalize(kc, config.kc_norm_post, training)

        if config.kc_dropout:
            kc = tf.layers.dropout(kc, config.kc_dropout_rate, training=training)

        if config.extra_layer:
            with tf.variable_scope('layer_extra', reuse=tf.AUTO_REUSE):
                n_neurons = config.extra_layer_neurons
                w3 = tf.get_variable('kernel', shape=(N_KC, n_neurons), dtype=tf.float32)
                # w3 = tf.abs(w3)
                b3 = tf.get_variable('bias', shape=(n_neurons,), dtype=tf.float32)
                kc = tf.nn.relu(tf.matmul(kc, w3) + b3)

        if config.kc_loss:
            # self.kc_loss = tf.reduce_mean(tf.pow(tf.abs(w_glo), 0.5)) * config.kc_loss_alpha
            self.kc_loss = tf.reduce_mean(tf.tanh(config.kc_loss_beta * w_glo)) * config.kc_loss_alpha
            self.loss += self.kc_loss

        if config.label_type == 'combinatorial':
            print('combinatorial')
            n_logits = config.n_combinatorial_classes
        else:
            n_logits = config.N_CLASS
        logits = tf.layers.dense(kc, n_logits, name='layer3', reuse=tf.AUTO_REUSE)

        if config.label_type == 'combinatorial':
            self.loss += tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
            pred = tf.cast(logits > .5, tf.int32)
            out = tf.reduce_all(tf.equal(pred, y), axis=1)
            self.acc = tf.reduce_mean(tf.to_float(out))
        elif config.label_type == 'one_hot':
            self.loss += tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            labels = tf.argmax(y, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, labels)))
        elif config.label_type == 'sparse':
            self.loss += tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
        elif config.label_type == 'multi_head_sparse':
            # second head
            logits2 = tf.layers.dense(kc, config.n_class_valence,
                                      name='layer3_2', reuse=tf.AUTO_REUSE)

            y1, y2 = tf.unstack(y, axis=1)

            loss1 = tf.losses.sparse_softmax_cross_entropy(
                labels=y1, logits=logits)
            loss2 = tf.losses.sparse_softmax_cross_entropy(
                labels=y2, logits=logits2)

            pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc1 = tf.reduce_mean(tf.to_float(tf.equal(pred1, y1)))
            pred2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)
            acc2 = tf.reduce_mean(tf.to_float(tf.equal(pred2, y2)))

            if config.train_head1:
                self.loss += loss1
            if config.train_head2:
                self.loss += loss2

            self.acc = acc1
            self.acc2 = acc2

        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")

        # print('USING L2 LOSS on ORN-PN weights!!')
        # self.loss += tf.reduce_sum(tf.square(w_orn)) * 0.01
        if config.receptor_layer:
            self.w_or = w_or

        self.w_orn = w_orn
        self.w_glo = w_glo
        self.glo_in = glo_in
        self.glo_in_pre = glo_in_pre
        self.glo = glo
        self.kc_in = kc_in
        self.kc = kc
        self.logits = logits

        self.pre_out = kc
        self.x = x

    def _build(self, x, y, training):
        config = self.config

        self.loss = 0
        orn = self._build_orn(x, training)
        pn = self._build_orn2pn(orn, training)
        kc = self._build_pn2kc(pn, training)

        if config.label_type == 'combinatorial':
            n_logits = config.N_COMBINATORIAL_CLASS
        else:
            n_logits = config.N_CLASS
        logits = tf.layers.dense(kc, n_logits, name='layer3', reuse=tf.AUTO_REUSE)

        if config.label_type == 'combinatorial':
            self.loss += tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        elif config.label_type == 'one_hot':
            self.loss += tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            labels = tf.argmax(y, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, labels)))
        elif config.label_type == 'sparse':
            self.loss += tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
        elif config.label_type == 'multi_head_sparse':
            # second head
            logits2 = tf.layers.dense(kc, config.n_class_valence,
                                      name='layer3_2', reuse=tf.AUTO_REUSE)

            y1, y2 = tf.unstack(y, axis=1)

            loss1 = tf.losses.sparse_softmax_cross_entropy(
                labels=y1, logits=logits)
            loss2 = tf.losses.sparse_softmax_cross_entropy(
                labels=y2, logits=logits2)

            pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc1 = tf.reduce_mean(tf.to_float(tf.equal(pred1, y1)))
            pred2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)
            acc2 = tf.reduce_mean(tf.to_float(tf.equal(pred2, y2)))

            if config.train_head1:
                self.loss += loss1
            if config.train_head2:
                self.loss += loss2

            self.acc = acc1
            self.acc2 = acc2

        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")

        # print('USING L2 LOSS on ORN-PN weights!!')
        # self.loss += tf.reduce_sum(tf.square(w_orn)) * 0.01

        self.kc = kc
        self.logits = logits

        self.pre_out = kc
        self.x = x

    def _build_orn(self, x, training):
        config = self.config
        N_PN = config.N_PN
        if config.receptor_layer:
            # Define another layer, layer0, that connects receptors to ORNs
            with tf.variable_scope('layer0', reuse= tf.AUTO_REUSE):
                N_OR = config.N_ORN
                ORN_DUP = config.N_ORN_DUPLICATION
                N_ORN = config.N_ORN * ORN_DUP
                range = 1/N_OR
                initializer = _initializer(range, config.initializer_or2orn)
                w_or = tf.get_variable('kernel', shape=(N_OR, N_ORN), dtype=tf.float32,
                                     initializer=initializer)
                if config.sign_constraint_or2orn:
                    w_or = tf.abs(w_or)

                if config.or2orn_normalization:
                    sums = tf.reduce_sum(w_or, axis=0)
                    w_or = tf.divide(w_or, sums)

                if config.or_bias:
                    b_or = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                            initializer=tf.constant_initializer(-0.01))
                else:
                    b_or = 0

                orn = tf.matmul(x, w_or) + b_or
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
                self.w_or = w_or
        else:
            if config.replicate_orn_with_tiling:
                # Replicating ORNs through tiling
                ORN_DUP = config.N_ORN_DUPLICATION
                N_ORN = config.N_ORN * ORN_DUP

                assert x.shape[-1] == config.N_ORN
                orn = tf.tile(x, [1, ORN_DUP])
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
            else:
                ORN_DUP = 1
                N_ORN = config.N_ORN
                orn = x
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)

        if config.orn_dropout:
            # This is interpreted as noise, so it's always on
            orn = tf.layers.dropout(orn, config.orn_dropout_rate,
                                    training=True)

        self.n_orn = N_ORN
        return orn

    def _build_orn2pn(self, orn, training):
        config = self.config
        N_PN = config.N_PN
        N_ORN = self.n_orn
        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            if config.sign_constraint_orn2pn:
                if config.direct_glo:
                    range = _sparse_range(config.N_ORN_DUPLICATION)
                else:
                    range = _sparse_range(N_ORN)
                initializer = _initializer(range, config.initializer_orn2pn)
                bias_initializer = tf.constant_initializer(0)
            else:
                initializer = tf.glorot_normal_initializer
                bias_initializer = tf.glorot_normal_initializer

            w1 = tf.get_variable('kernel', shape=(N_ORN, N_PN),
                                 dtype=tf.float32,
                                 initializer=initializer)

            b_orn = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                    initializer= bias_initializer)

            if config.direct_glo:
                mask = np.tile(np.eye(N_PN), (config.N_ORN_DUPLICATION, 1)) / config.N_ORN_DUPLICATION
                w_orn = w1 * mask
            else:
                w_orn = w1

            if config.sign_constraint_orn2pn:
                w_orn = tf.abs(w_orn)

            if config.orn2pn_normalization:
                sums = tf.reduce_sum(w_orn, axis=0, keepdims=True)
                w_orn = tf.divide(w_orn, sums)

            glo_in_pre = tf.matmul(orn, w_orn) + b_orn
            if config.skip_orn2pn:
                glo_in = orn
            else:
                glo_in = _normalize(glo_in_pre, config.pn_norm_pre, training)

            # self.glo_in_pre_mean = tf.reduce_mean(glo_in_pre, axis=1)
            # self.glo_in_mean = tf.reduce_mean(glo_in, axis=1)

            glo = tf.nn.relu(glo_in)
            glo = _normalize(glo, config.pn_norm_post, training)

        self.w_orn = w_orn
        self.glo_in = glo_in
        self.glo_in_pre = glo_in_pre
        self.glo = glo

        return glo

    def _build_pn2kc(self, pn, training):
        config = self.config
        N_KC = config.N_KC
        N_PN = config.N_PN
        N_ORN = self.n_orn
        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            if config.skip_orn2pn:
                N_USE = N_ORN
            else:
                N_USE = N_PN

            if config.sign_constraint_pn2kc:
                if config.initial_pn2kc == 0:
                    if config.sparse_pn2kc:
                        range = _sparse_range(config.kc_inputs)
                    else:
                        range = _sparse_range(N_USE)
                else:
                    range = config.initial_pn2kc
                initializer = _initializer(range, config.initializer_pn2kc)
                bias_initializer = tf.constant_initializer(config.kc_bias)
            else:
                initializer = tf.glorot_normal_initializer
                bias_initializer = tf.glorot_normal_initializer

            w2 = tf.get_variable('kernel', shape=(N_USE, N_KC), dtype=tf.float32,
                                 initializer= initializer)
            b_glo = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer= bias_initializer)

            if config.sparse_pn2kc:
                if config.skip_orn2pn:
                    w_mask = get_sparse_mask(N_USE, N_KC, config.kc_inputs, complex=True)
                else:
                    w_mask = get_sparse_mask(N_USE, N_KC, config.kc_inputs)
                w_mask = tf.get_variable(
                    'mask', shape=(N_USE, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_mask),
                    trainable=False)
                w_glo = tf.multiply(w2, w_mask)
            else:
                w_glo = w2

            if config.sign_constraint_pn2kc:
                w_glo = tf.abs(w_glo)

            if config.mean_subtract_pn2kc:
                w_glo -= tf.reduce_mean(w_glo, axis=0)

            # KC input before activation function
            kc_in = tf.matmul(pn, w_glo) + b_glo
            kc_in = _normalize(kc_in, config.kc_norm_pre, training)
            if 'skip_pn2kc' in dir(config) and config.skip_pn2kc:
                kc_in = pn
            kc = tf.nn.relu(kc_in)
            kc = _normalize(kc, config.kc_norm_post, training)

        if 'apl' in dir(config) and config.apl:
            if config.skip_pn2kc:
                raise ValueError('apl can not be used when no KC.')
            with tf.variable_scope('kc2apl', reuse=tf.AUTO_REUSE):
                w_kc2apl0 = tf.get_variable(
                    'kernel', shape=(N_KC, 1), dtype=tf.float32)
                b_apl = tf.get_variable('bias', shape=(1,), dtype=tf.float32)
                w_kc2apl = tf.abs(w_kc2apl0)

                apl = tf.nn.relu(tf.matmul(kc, w_kc2apl) + b_apl)

            with tf.variable_scope('apl2kc', reuse=tf.AUTO_REUSE):
                w_apl2kc0 = tf.get_variable(
                    'kernel', shape=(1, N_KC), dtype=tf.float32)
                w_apl2kc = - tf.abs(w_apl2kc0)  # inhibitory connections

            kc = tf.nn.relu(tf.matmul(apl, w_apl2kc) + kc_in)

        if config.kc_dropout:
            kc = tf.layers.dropout(kc, config.kc_dropout_rate, training=training)

        if config.extra_layer:
            with tf.variable_scope('layer_extra', reuse=tf.AUTO_REUSE):
                n_neurons = config.extra_layer_neurons
                w3 = tf.get_variable('kernel', shape=(N_KC, n_neurons), dtype=tf.float32)
                # w3 = tf.abs(w3)
                b3 = tf.get_variable('bias', shape=(n_neurons,), dtype=tf.float32)
                kc = tf.nn.relu(tf.matmul(kc, w3) + b3)

        if config.kc_loss:
            # self.kc_loss = tf.reduce_mean(tf.pow(tf.abs(w_glo), 0.5)) * config.kc_loss_alpha
            self.kc_loss = tf.reduce_mean(tf.tanh(config.kc_loss_beta * w_glo)) * config.kc_loss_alpha
            self.loss += self.kc_loss

        self.w_glo = w_glo
        self.kc_in = kc_in

        return kc

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, 'model.pkl')

        sess = tf.get_default_session()
        var_dict = {v.name: sess.run(v) for v in tf.trainable_variables()}
        if self.config.receptor_layer:
            var_dict['w_or'] = sess.run(self.w_or)
            var_dict['w_combined'] = np.matmul(sess.run(self.w_or), sess.run(self.w_orn))
        var_dict['w_orn'] = sess.run(self.w_orn)
        var_dict['w_glo'] = sess.run(self.w_glo)
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)

    def set_oracle_weights(self):
        """Set the weights to be prototype matching oracle weights."""
        config = self.config
        sess = tf.get_default_session()
        prototype = np.load(os.path.join(config.data_dir, 'prototype.npy'))
        # Connection weights
        prototype_repr = sess.run(self.pre_out, {self.x: prototype})
        w_oracle, b_oracle = _get_oracle(prototype_repr)
        w_oracle *= config.oracle_scale
        b_oracle *= config.oracle_scale

        w_out = [v for v in tf.trainable_variables() if
                 v.name == 'model/layer3/kernel:0'][0]
        b_out = [v for v in tf.trainable_variables() if
                 v.name == 'model/layer3/bias:0'][0]

        sess.run(w_out.assign(w_oracle))
        sess.run(b_out.assign(b_oracle))

    def perturb_weights(self, scale):
        """Perturb all weights with multiplicative noise.

        Args:
            scale: float. Perturb weights with
                random variables ~ U[1-scale, 1+scale]
        """
        sess = tf.get_default_session()

        def perturb(w):
            w = w * np.random.uniform(1-scale, 1+scale, size=w.shape)
            return w

        # record original weight values when perturb for the first time
        if not hasattr(self, 'origin_weights'):
            print('Perturbing weights:')
            for v in tf.trainable_variables():
                print(v)
            self.origin_weights = [sess.run(v) for v in tf.trainable_variables()]

        for v_value, v in zip(self.origin_weights, tf.trainable_variables()):
            sess.run(v.assign(perturb(v_value)))



def _signed_dense(x, n0, n1, training):
    w1 = tf.get_variable('kernel', shape=(n0, n1), dtype=tf.float32)
    b_orn = tf.get_variable('bias', shape=(n1,), dtype=tf.float32,
                            initializer=tf.zeros_initializer())

    w_orn = tf.abs(w1)
    # w_orn = w1
    glo_in_pre = tf.matmul(x, w_orn) + b_orn
    glo_in = _normalize(glo_in_pre, 'batch_norm', training)
    # glo_in = _normalize(glo_in_pre, None, training)
    glo = tf.nn.relu(glo_in)
    return glo

class RNN(Model):

    def __init__(self, x, y, config=None, training=True):

        if config is None:
            config = FullConfig
        self.config = config

        super(RNN, self).__init__(config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, training)

        if training:
            optimizer = tf.train.AdamOptimizer(config.lr)

            var_list = tf.trainable_variables()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, training):
        config = self.config
        ORN_DUP = config.N_ORN_DUPLICATION
        N_ORN = config.N_ORN * ORN_DUP
        NOISE = config.ORN_NOISE_STD
        NEURONS = N_ORN + config.NEURONS
        TIME_STEPS = config.TIME_STEPS

        # Replicating ORNs through tiling
        assert x.shape[-1] == config.N_ORN
        x = tf.tile(x, [1, ORN_DUP])
        x += tf.random_normal(x.shape, stddev=NOISE)

        W_in_np = np.zeros([N_ORN, NEURONS])
        np.fill_diagonal(W_in_np, 1)
        W_in = tf.constant(W_in_np, dtype=tf.float32, name='W_in')
        rnn_output = tf.matmul(x, W_in)

        rnn_outputs = []
        rnn_outputs.append(rnn_output)
        with tf.variable_scope('layer_rnn', reuse=tf.AUTO_REUSE):
            # TODO: do not want ORNs to connect to each other, nor for them to have a bias
            initializer = _initializer(_sparse_range(config.N_ORN), arg='constant')
            w_rnn = tf.get_variable('kernel', shape=(NEURONS, NEURONS), dtype=tf.float32, initializer=initializer)
            w_rnn = tf.abs(w_rnn)
            b_rnn = tf.get_variable('bias', shape=NEURONS, dtype=tf.float32, initializer=tf.constant_initializer(-1))
            for t in range(TIME_STEPS):
                rnn_output = tf.matmul(rnn_output, w_rnn) + b_rnn
                rnn_output = tf.nn.relu(rnn_output)
                rnn_outputs.append(rnn_output)

        if config.BATCH_NORM:
            rnn_output = _normalize(rnn_output, 'batch_norm', training)

        if config.dropout:
            rnn_output = tf.layers.dropout(rnn_output, config.dropout_rate, training=training)

        # logits = tf.layers.dense(kc, n_logits, name='layer_out', reuse=tf.AUTO_REUSE)
        with tf.variable_scope('layer_out', reuse=tf.AUTO_REUSE):
            # TODO: do not want ORNs to output to classes
            initializer = _initializer(_sparse_range(config.N_ORN), arg='uniform')
            w_out = tf.get_variable('kernel', shape=(NEURONS, config.N_CLASS), dtype=tf.float32)
            w_out = tf.abs(w_out)
            b_out = tf.get_variable('bias', shape=config.N_CLASS, dtype=tf.float32)
            logits = tf.matmul(rnn_output, w_out) + b_out

        # debug
        # w_rnn = tf.get_variable('kernel_', shape=(N_ORN, NEURONS), dtype=tf.float32)
        # w_rnn = tf.abs(w_rnn)
        # b_rnn = tf.get_variable('bias_', shape=NEURONS, dtype=tf.float32)
        # w_out = tf.get_variable('kernel', shape=(NEURONS, config.N_CLASS), dtype=tf.float32)
        # b_out = tf.get_variable('bias', shape=config.N_CLASS, dtype=tf.float32)
        # rnn_output = tf.matmul(x, w_rnn) + b_rnn
        # rnn_output = tf.nn.relu(rnn_output)
        # rnn_output = tf.layers.dropout(rnn_output, config.dropout_rate, training=training)
        # logits = tf.matmul(rnn_output, w_out) + b_out


        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=logits)
        if config.WEIGHT_LOSS:
            loss += config.WEIGHT_ALPHA * tf.reduce_mean(tf.tanh(w_rnn))
            # loss += config.WEIGHT_ALPHA * tf.reduce_mean(tf.tanh(w_out))

        self.loss = loss
        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))

        self.logits = logits
        self.w_rnn = w_rnn
        self.b_rnn = b_rnn
        self.w_out = w_out
        self.b_out = b_out
        self.rnn_outputs = rnn_outputs

    def set_weights(self):
        """Set the weights to be prototype matching oracle weights."""
        sess = tf.get_default_session()

        w_rnn_tf = [v for v in tf.trainable_variables() if
                 v.name == 'model/layer_rnn/kernel:0'][0]
        w_rnn_values = sess.run(w_rnn_tf)
        np.fill_diagonal(w_rnn_values, 1)
        sess.run(w_rnn_tf.assign(w_rnn_values))

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, 'model.pkl')
        sess = tf.get_default_session()
        var_dict = {v.name: sess.run(v) for v in tf.trainable_variables()}
        var_dict['w_rnn'] = sess.run(self.w_rnn)
        var_dict['b_rnn'] = sess.run(self.b_rnn)
        var_dict['w_out'] = sess.run(self.w_out)
        var_dict['b_out'] = sess.run(self.b_out)
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)



class NormalizedMLP(Model):
    """Normalized multi-layer perceptron model.

    This model is simplified compared to the full model, with fewer options available
    """

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

        super(NormalizedMLP, self).__init__(config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, training)

        if training:
            optimizer = tf.train.AdamOptimizer(config.lr)
            var_list = tf.trainable_variables()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)
            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, training):
        config = self.config
        ORN_DUP = config.N_ORN_DUPLICATION
        N_ORN = config.N_ORN * ORN_DUP
        NEURONS = [N_ORN] + list(config.NEURONS)
        n_layer = len(config.NEURONS)  # number of hidden layers

        # Replicating ORNs through tiling
        assert x.shape[-1] == config.N_ORN
        x = tf.tile(x, [1, ORN_DUP])
        x += tf.random_normal(x.shape, stddev=config.ORN_NOISE_STD)

        if config.orn_dropout:
            x = tf.layers.dropout(x, config.orn_dropout_rate,
                                    training=True)

        y_hat = x
        for i_layer in range(n_layer):
            layername = 'layer' + str(i_layer+1)
            with tf.variable_scope(layername, reuse=tf.AUTO_REUSE):
                    y_hat = _signed_dense(
                        y_hat, NEURONS[i_layer], NEURONS[i_layer+1], training)

        layername = 'layer' + str(n_layer + 1)
        logits = tf.layers.dense(y_hat, config.N_CLASS,
                                 name=layername, reuse=tf.AUTO_REUSE)

        self.loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=logits)

        pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
        self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
        self.logits = logits


class AutoEncoder(Model):
    """Simple autoencoder network."""

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

        super(AutoEncoder, self).__init__(config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, training)

        if training:
            optimizer = tf.train.AdamOptimizer(config.lr)

            var_list = tf.trainable_variables()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, training):
        config = self.config
        N_KC = config.N_KC
        n_orn = config.n_orn

        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            if config.initial_pn2kc == 0:
                if config.sparse_pn2kc:
                    range = _sparse_range(config.kc_inputs)
                else:
                    range = _sparse_range(n_orn)
            else:
                range = config.initial_pn2kc

            initializer = _initializer(range, config.initializer_pn2kc)
            w2 = tf.get_variable('kernel', shape=(config.n_orn, N_KC), dtype=tf.float32,
                                 initializer= initializer)

            b_glo = tf.get_variable('bias', shape=(N_KC,), dtype=tf.float32,
                                    initializer=tf.constant_initializer(config.kc_bias))

            if config.sparse_pn2kc:
                w_mask = get_sparse_mask(n_orn, N_KC, config.kc_inputs)
                w_mask = tf.get_variable(
                    'mask', shape=(n_orn, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(w_mask),
                    trainable=False)
                w_glo = tf.multiply(w2, w_mask)
            else:
                w_glo = w2

            if config.sign_constraint_pn2kc:
                w_glo = tf.abs(w_glo)
                # w_glo = tf.nn.sigmoid(w_glo)
                # w_glo = tf.nn.softplus(w_glo)
                # w_glo = tf.nn.relu(w_glo)

            if config.mean_subtract_pn2kc:
                w_glo -= tf.reduce_mean(w_glo, axis=0)

            # KC input before activation function
            kc_in = tf.matmul(x, w_glo) + b_glo
            kc_in = _normalize(kc_in, config.kc_norm_pre, training)
            kc = tf.nn.relu(kc_in)
            kc = _normalize(kc, config.kc_norm_post, training)

        logits = tf.layers.dense(kc, config.n_orn, name='layer3')

        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        # self.loss = tf.reduce_mean(tf.square(y - logits))

        pred = tf.to_float(tf.round(tf.sigmoid(logits)))

        self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))

        self.w_glo = w_glo

    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fname = os.path.join(save_path, 'model.pkl')

        sess = tf.get_default_session()
        var_dict = {v.name: sess.run(v) for v in tf.trainable_variables()}
        var_dict['w_glo'] = sess.run(self.w_glo)
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)


class AutoEncoderSimple(Model):
    """Simple autoencoder network."""

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

        super(AutoEncoderSimple, self).__init__(config.save_path)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self._build(x, y, training)

        if training:
            optimizer = tf.train.AdamOptimizer(config.lr)

            var_list = tf.trainable_variables()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver()

    def _build(self, x, y, training):
        config = self.config
        N_KC = config.N_KC
        n_orn = config.n_orn


        # KC input before activation function
        kc = tf.layers.dense(x, config.N_KC, name='layer2')
        kc = tf.nn.relu(kc)

        logits = tf.layers.dense(kc, config.n_orn, name='layer3',
                                 kernel_initializer=tf.zeros_initializer)

        logits = logits + x

        self.loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        # self.loss = tf.reduce_mean(tf.square(y - logits))

        pred = tf.to_float(tf.round(tf.sigmoid(logits)))

        self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
