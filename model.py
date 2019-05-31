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
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
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

    def lesion_units(self, name, units, verbose=False, arg='outbound'):
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
        var_lesion = [tmp for tmp in tf.trainable_variables() if tmp.name == name]
        if var_lesion:
            v = var_lesion[0]
        else:
            print('No units are being lesioned')
            return
        # Connection weights
        v_val = sess.run(v)

        if arg == 'outbound':
            v_val[units, :] = 0
        elif arg == 'inbound':
            v_val[:,units] = 0
        else:
            raise ValueError('did not recognize lesion argument: {}'.format(arg))

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
            outputs = tf.contrib.layers.layer_norm(
                inputs, center=True, scale=True)
            # outputs = tf.contrib.layers.layer_norm(
            #     inputs, center=True, scale=False)
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
            r_max = tf.get_variable('r_max', shape=(1, 50), dtype=tf.float32, initializer=tf.constant_initializer(25))
            rho = tf.get_variable('rho', shape=(1,50), dtype=tf.float32, initializer=tf.constant_initializer(1))
            m = tf.get_variable('m', shape=(1,50), dtype=tf.float32, initializer=tf.constant_initializer(0.01))
            sums = tf.reduce_sum(inputs, axis=1, keepdims=True) + 1e-6
            num = r_max * tf.pow(inputs, exp)
            den = tf.pow(inputs, exp) + rho + tf.pow(m * sums, exp)
            outputs =  tf.divide(num, den)
        elif norm_type == 'activity':
            r_max = tf.get_variable('r_max', shape=(1, 50), dtype=tf.float32, initializer=tf.constant_initializer(100))
            sums = tf.reduce_sum(inputs, axis=1, keepdims=True) + 1e-6
            outputs = r_max * tf.divide(inputs, sums)
        elif norm_type == 'fixed_activity':
            r_max = 25
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

    def __init__(self, x, y, config=None, training=True, meta_learn = False):
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
        self.weights = dict()

        super(FullModel, self).__init__(self.config.save_path)

        if meta_learn == False:
            self._build(x, y, training)
            if training:
                optimizer = tf.train.AdamOptimizer(self.config.lr)
                # optimizer = tf.train.AdagradOptimizer(self.config.lr)
                # optimizer = tf.train.GradientDescentOptimizer(self.config.lr)
                # optimizer = tf.train.RMSPropOptimizer(self.config.lr)

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

            self.saver = tf.train.Saver(max_to_keep=None)
        # self.saver = tf.train.Saver(tf.trainable_variables())

    def loss_func(self, logits, logits2, y):
        valence_loss_coeff = 1
        config = self.config
        class_loss = 0
        if config.label_type == 'combinatorial':
            class_loss += tf.losses.sigmoid_cross_entropy(multi_class_labels=y, logits=logits)
        elif config.label_type == 'one_hot':
            class_loss += tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits)
        elif config.label_type == 'sparse':
            class_loss += tf.losses.sparse_softmax_cross_entropy(labels=y,
                                                           logits=logits)
        elif config.label_type == 'multi_head_sparse':
            y1, y2 = tf.unstack(y, axis=1)
            loss1 = tf.losses.sparse_softmax_cross_entropy(
                labels=y1, logits=logits)
            loss2 = tf.losses.sparse_softmax_cross_entropy(
                labels=y2, logits=logits2)
            if config.train_head1:
                class_loss += loss1
            if config.train_head2:
                class_loss += valence_loss_coeff * loss2
        elif config.label_type == 'multi_head_one_hot':
            y1 = y[:,:self.config.N_CLASS]
            y2 = y[:, self.config.N_CLASS:]
            loss1 = tf.losses.softmax_cross_entropy(onehot_labels=y1, logits=logits)
            loss2 = tf.losses.softmax_cross_entropy(onehot_labels=y2, logits=logits2)
            if config.train_head1:
                class_loss += loss1
            if config.train_head2:
                class_loss += valence_loss_coeff * loss2
        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")
        self.loss = class_loss
        return class_loss

    def accuracy_func(self, logits, logits2, y):
        config = self.config
        self.acc2 = tf.constant(0, dtype=tf.float32)
        if config.label_type == 'combinatorial':

            self.acc = tf.contrib.metrics.streaming_pearson_correlation(
                predictions = tf.math.sigmoid(logits), labels= y)[1]
        elif config.label_type == 'one_hot':
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            labels = tf.argmax(y, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, labels)))
        elif config.label_type == 'sparse':
            pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
            self.acc = tf.reduce_mean(tf.to_float(tf.equal(pred, y)))
        elif config.label_type == 'multi_head_sparse':
            y1, y2 = tf.unstack(y, axis=1)
            pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
            acc1 = tf.reduce_mean(tf.to_float(tf.equal(pred1, y1)))
            pred2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)
            acc2 = tf.reduce_mean(tf.to_float(tf.equal(pred2, y2)))
            self.acc = acc1
            self.acc2 = acc2
        elif config.label_type == 'multi_head_one_hot':
            y1 = y[:,:self.config.N_CLASS]
            y2 = y[:, self.config.N_CLASS:]
            pred1 = tf.argmax(logits, axis=-1, output_type=tf.int32)
            labels1 = tf.argmax(y1, axis=-1, output_type=tf.int32)
            acc1 = tf.reduce_mean(tf.to_float(tf.equal(pred1, labels1)))
            pred2 = tf.argmax(logits2, axis=-1, output_type=tf.int32)
            labels2 = tf.argmax(y2, axis=-1, output_type=tf.int32)
            acc2 = tf.reduce_mean(tf.to_float(tf.equal(pred2, labels2)))
            self.acc = acc1
            self.acc2 = acc2

        else:
            raise ValueError("""labels are in any of the following formats:
                                combinatorial, one_hot, sparse""")
        return (self.acc, self.acc2)


    def _build(self, x, y, training):
        with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
            self.build_weights()
            logits, logits2 = self.build_activity(x, self.weights, training)
            loss = self.loss_func(logits=logits, logits2= logits2, y=y)
            acc = self.accuracy_func(logits= logits, logits2=logits2, y=y)

    def build_weights(self):
        self._build_or2orn_weights()
        self._build_orn2pn_weights()
        self._build_pn2kc_weights()
        self._build_kc2logit_weights()
        return self.weights

    def build_activity(self, x, weights, training, reuse=True):
        orn = self._build_orn_activity(x, weights, training)
        pn = self._build_pn_activity(orn, weights, training)
        if 'apl' in dir(self.config) and self.config.apl:
            kc = self._build_kc_activity_withapl(pn, weights, training)
        else:
            kc = self._build_kc_activity(pn, weights, training)
        logits, logits2 = self._build_logit_activity(kc, weights, training)
        return logits, logits2

    def _build_or2orn_weights(self):
        config = self.config
        N_OR = config.N_ORN
        ORN_DUP = config.N_ORN_DUPLICATION
        if config.receptor_layer:
            N_ORN = config.N_ORN * ORN_DUP
            with tf.variable_scope('layer0', reuse=tf.AUTO_REUSE):
                range = 1 / N_OR
                initializer = _initializer(range, config.initializer_or2orn)
                w_or = tf.get_variable('kernel', shape=(N_OR, N_ORN), dtype=tf.float32,
                                       initializer=initializer)
                if config.sign_constraint_or2orn:
                    w_or = tf.abs(w_or)

                if config.or2orn_normalization:
                    sums = tf.reduce_sum(w_or, axis=0)
                    w_or = tf.divide(w_or, sums)

                if config.or_bias:
                    b_or = tf.get_variable('bias', shape=(N_OR,), dtype=tf.float32,
                                           initializer=tf.constant_initializer(-0.01))
                else:
                    b_or = 0
                self.weights['w_or'] = w_or
                self.weights['b_or'] = b_or
                self.w_or = w_or
        else:
            if config.replicate_orn_with_tiling:
                N_ORN = N_OR * ORN_DUP
            else:
                N_ORN = N_OR
        self.n_orn = N_ORN

    def _build_orn2pn_weights(self):
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
                initializer = tf.glorot_uniform_initializer()
                bias_initializer = tf.zeros_initializer()

            w_orn = tf.get_variable('kernel', shape=(N_ORN, N_PN),
                                 dtype=tf.float32,
                                 initializer=initializer)

            b_orn = tf.get_variable('bias', shape=(N_PN,), dtype=tf.float32,
                                    initializer= bias_initializer)
            if config.sign_constraint_orn2pn:
                w_orn = tf.abs(w_orn)
        self.weights['w_orn'] = w_orn
        self.weights['b_orn'] = b_orn
        self.w_orn = w_orn

    def _build_pn2kc_weights(self):
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

            if 'equal_kc_bias' in dir(config) and config.equal_kc_bias:
                b_glo = tf.get_variable('bias', shape=(), dtype=tf.float32,
                                        initializer=tf.zeros_initializer())
            else:
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

        if 'apl' in dir(config) and config.apl:
            if config.skip_pn2kc:
                raise ValueError('apl can not be used when no KC.')
            with tf.variable_scope('kc2apl', reuse=tf.AUTO_REUSE):
                w_kc2apl0 = tf.get_variable(
                    'kernel', shape=(N_KC, 1), dtype=tf.float32,
                    initializer=tf.constant_initializer(1./N_KC))
                b_apl = tf.get_variable('bias', shape=(1,), dtype=tf.float32)
                w_kc2apl = tf.abs(w_kc2apl0)

            with tf.variable_scope('apl2kc', reuse=tf.AUTO_REUSE):
                w_apl2kc0 = tf.get_variable(
                    'kernel', shape=(1, N_KC), dtype=tf.float32,
                    initializer=tf.constant_initializer(0.1)
                )
                w_apl2kc = - tf.abs(w_apl2kc0)  # inhibitory connections

            # with tf.variable_scope('apl', reuse=tf.AUTO_REUSE):
            #     # w_apl = tf.get_variable('kernel', shape=(1,), dtype=tf.float32,
            #     #                         initializer=tf.constant_initializer(1.))
            #     w_apl = 2.0
            #     apl_in = tf.abs(w_apl) * tf.reduce_mean(kc, axis=1, keepdims=True)
            #     kc = tf.nn.relu(kc_in - apl_in)

            self.weights['w_apl_in'] = w_kc2apl
            self.weights['w_apl_out'] = w_apl2kc
            self.weights['b_apl'] = b_apl

        if config.kc_loss:
            self.kc_loss = tf.reduce_mean(tf.tanh(config.kc_loss_beta * w_glo)) * config.kc_loss_alpha
            self.loss += self.kc_loss

        if config.extra_layer:
            with tf.variable_scope('layer_extra', reuse=tf.AUTO_REUSE):
                n_neurons = config.extra_layer_neurons
                w3 = tf.get_variable('kernel', shape=(N_KC, n_neurons), dtype=tf.float32)
                b3 = tf.get_variable('bias', shape=(n_neurons,), dtype=tf.float32)
                self.weights['w_extra_layer'] = w3
                self.weights['b_extra_layer'] = b3

        self.weights['w_glo'] = w_glo
        self.weights['b_glo'] = b_glo
        self.w_glo = w_glo

    def _build_kc2logit_weights(self):
        config = self.config
        if config.label_type == 'combinatorial':
            n_logits = config.n_combinatorial_classes
        else:
            n_logits = config.N_CLASS

        with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
            if config.skip_pn2kc:
                input_size = config.N_PN
            else:
                input_size = config.N_KC
            w_output = tf.get_variable(
                'kernel', shape=(input_size, n_logits),
                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            self.weights['w_output'] = w_output

            if 'output_bias' not in dir(config) or config.output_bias:
                b_output = tf.get_variable(
                    'bias', shape=(n_logits,), dtype=tf.float32,
                    initializer=tf.zeros_initializer())
                self.weights['b_output'] = b_output

        if config.label_type == 'multi_head_sparse' or config.label_type == 'multi_head_one_hot':
            with tf.variable_scope('layer3_2', reuse=tf.AUTO_REUSE):
                w_output = tf.get_variable(
                    'kernel', shape=(config.N_KC, config.n_class_valence),
                    dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
                b_output = tf.get_variable(
                    'bias', shape=(config.n_class_valence,), dtype=tf.float32,
                    initializer=tf.zeros_initializer())
                self.weights['w_output_head2'] = w_output
                self.weights['b_output_head2'] = b_output

    def _build_orn_activity(self, x, weights, training):
        config = self.config
        ORN_DUP = config.N_ORN_DUPLICATION
        if config.receptor_layer:
            with tf.variable_scope('layer0', reuse=tf.AUTO_REUSE):
                w_or = weights['w_or']
                b_or = weights['b_or']
                orn = tf.matmul(x, w_or) + b_or
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
        else:
            if config.replicate_orn_with_tiling:
                # Replicating ORNs through tiling
                assert x.shape[-1] == config.N_ORN
                orn = tf.tile(x, [1, ORN_DUP])
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)
            else:
                orn = x
                orn = _noise(orn, config.NOISE_MODEL, config.ORN_NOISE_STD)

        orn = _normalize(orn, config.orn_norm, training)
        if config.orn_dropout:
            # This is interpreted as noise, so it's always on
            orn = tf.layers.dropout(orn, config.orn_dropout_rate, training=True)
        self.x = x
        return orn

    def _build_pn_activity(self, orn, weights, training):
        config = self.config
        w_orn = weights['w_orn']
        b_orn = weights['b_orn']
        N_PN = config.N_PN
        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            glo_in_pre = tf.matmul(orn, w_orn) + b_orn
            if config.skip_orn2pn:
                glo_in = orn
            elif config.direct_glo:
                mask = np.tile(np.eye(N_PN), (config.N_ORN_DUPLICATION, 1)) / config.N_ORN_DUPLICATION
                glo_in = tf.matmul(orn, mask.astype(np.float32))
                glo_in = _normalize(glo_in, config.pn_norm_pre, training)
            else:
                glo_in = _normalize(glo_in_pre, config.pn_norm_pre, training)
            glo = tf.nn.relu(glo_in)
            glo = _normalize(glo, config.pn_norm_post, training)
        self.glo_in = glo_in
        self.glo_in_pre = glo_in_pre
        self.glo = glo
        return glo

    def _build_kc_activity(self, pn, weights, training):
        # KC input before activation function
        config = self.config
        w_glo = weights['w_glo']
        b_glo = weights['b_glo']

        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            kc_in = tf.matmul(pn, w_glo) + b_glo
            kc_in = _normalize(kc_in, config.kc_norm_pre, training)
            if 'skip_pn2kc' in dir(config) and config.skip_pn2kc:
                kc_in = pn

            kc = tf.nn.relu(kc_in)
            kc = _normalize(kc, config.kc_norm_post, training)

            if config.kc_dropout:
                kc = tf.layers.dropout(kc, config.kc_dropout_rate, training=training)

            if config.extra_layer:
                w3 = weights['w_extra_layer']
                b3 = weights['b_extra_layer']
                kc = tf.nn.relu(tf.matmul(kc, w3) + b3)
        self.kc_in = kc_in
        self.kc = kc
        return kc

    def _build_kc_activity_withapl(self, pn, weights, training):
        # KC input before activation function
        config = self.config
        w_glo = weights['w_glo']
        b_glo = weights['b_glo']

        kc_in = tf.matmul(pn, w_glo) + b_glo
        kc = tf.nn.relu(kc_in)

        # kc_in = tf.matmul(pn, w_glo)
        # kc = tf.nn.relu(kc_in + b_glo)

        w_kc2apl = weights['w_apl_in']
        b_apl = weights['b_apl']
        w_apl2kc = weights['w_apl_out']

        # sigmoidal APL with subtractive inhibition
        # apl = tf.nn.sigmoid(tf.matmul(kc, w_kc2apl) + b_apl)  # standard
        # kc_in = tf.matmul(apl, w_apl2kc) + kc_in

        # multiplicative APL inhibition
        apl = tf.nn.relu(tf.matmul(kc, w_kc2apl) + b_apl)
        kc_in = kc_in * tf.nn.sigmoid(tf.matmul(apl, w_apl2kc))
        # kc_in = kc_in / (1 - tf.matmul(apl, w_apl2kc))

        kc_in = _normalize(kc_in, config.kc_norm_pre, training)
        kc = tf.nn.relu(kc_in)
        # kc = tf.nn.relu(kc_in + b_glo)

        if config.kc_dropout:
            kc = tf.layers.dropout(kc, config.kc_dropout_rate, training=training)

        self.kc_in = kc_in
        self.kc = kc
        return kc

    def _build_logit_activity(self, kc, weights, training):
        config = self.config

        with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
            logits = tf.matmul(kc, weights['w_output'])
            if 'output_bias' not in dir(config) or config.output_bias:
                logits = logits + weights['b_output']

            if config.label_type == 'multi_head_sparse' or config.label_type == 'multi_head_one_hot':
                logits2= tf.matmul(kc, weights['w_output_head2']) + weights['b_output_head2']
            else:
                logits2 = tf.constant(0, dtype=tf.float32)

        self.logits = logits
        self.logits2 = logits2
        return logits, logits2


    def save_pickle(self, epoch=None):
        """Save model using pickle.

        This is quite space-inefficient. But it's easier to read out.
        """
        save_path = self.save_path
        if epoch is not None:
            save_path = os.path.join(save_path, 'epoch', str(epoch).zfill(4))
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
        prototype_repr = sess.run(self.kc, {self.x: prototype})
        w_oracle, b_oracle = _get_oracle(prototype_repr)
        w_oracle *= config.oracle_scale
        b_oracle *= config.oracle_scale

        w_out = [v for v in tf.trainable_variables() if
                 v.name == 'model/layer3/kernel:0'][0]
        b_out = [v for v in tf.trainable_variables() if
                 v.name == 'model/layer3/bias:0'][0]

        sess.run(w_out.assign(w_oracle))
        sess.run(b_out.assign(b_oracle))


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
            # optimizer = tf.train.GradientDescentOptimizer(config.lr)
            optimizer = tf.train.AdamOptimizer(config.lr)

            var_list = tf.trainable_variables()

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(self.loss, var_list=var_list)

            print('Training variables')
            for v in var_list:
                print(v)

        self.saver = tf.train.Saver(max_to_keep=None)

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

        loss = tf.losses.sparse_softmax_cross_entropy(
            labels=y, logits=logits)
        if config.WEIGHT_LOSS:
            loss += config.WEIGHT_ALPHA * tf.reduce_mean(tf.tanh(w_rnn))

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

        self.saver = tf.train.Saver(max_to_keep=None)

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

        self.saver = tf.train.Saver(max_to_keep=None)

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

        self.saver = tf.train.Saver(max_to_keep=None)

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
