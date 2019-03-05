""" Code for the MAML algorithm and network definitions.

Adpated from Chelsea Finn's code
"""
from __future__ import print_function

import os
import pickle

import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

from model import Model

FLAGS = flags.FLAGS

def normalize(inp, activation, reuse, scope):
    if FLAGS.norm == 'batch_norm':
        return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'layer_norm':
        return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
    elif FLAGS.norm == 'None':
        if activation is not None:
            return activation(inp)
        else:
            return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1])
    label = tf.reshape(label, [-1])
    return tf.reduce_mean(tf.square(pred-label))


def xent(pred, label):
    return tf.losses.softmax_cross_entropy(onehot_labels=label, logits=pred)


def acc_func(pred, label):
    return tf.reduce_mean(tf.to_float(tf.equal(
        tf.argmax(pred, 1), tf.argmax(label, 1))))


class MAML:
    def __init__(self, x, y, config, test_num_updates=5):
        """MAML model."""
        self.test_num_updates = test_num_updates
        self.model = PNKCModel(config)
        self.loss_func = xent
        # self.loss_func = mse
        self.save_pickle = self.model.save_pickle

        self._build(x, y)

    def task_metalearn(self, inp, reuse=True):
        """ Perform gradient descent for one task in the meta-batch.

        Args:
            inp: a sequence unpacked to inputa, inputb, labela, labelb
            inputa: tensor (batch_size, dim_input)

        Returns:
            task_output: a sequence unpacked to outputa, outputb, lossa, lossb
        """
        weights = self.weights
        num_updates = max(self.test_num_updates, FLAGS.num_updates)
        inputa, inputb, labela, labelb = inp
        task_outputbs, task_lossesb, task_accuraciesb = [], [], []

        task_outputa = self.model.build(inputa, weights,
                                        reuse=reuse)  # only reuse on the first iter
        task_lossa = self.loss_func(task_outputa, labela)
        task_accuracya = acc_func(task_outputa, labela)

        grads = tf.gradients(task_lossa, list(weights.values()))
        if FLAGS.stop_grad:
            grads = [tf.stop_gradient(grad) for grad in grads]

        # manually construct the weights post inner gradient descent
        # Notice that this doesn't have to be through gradient descent
        gradients = dict(zip(weights.keys(), grads))
        fast_weights = dict()
        for key in weights.keys():
            if key in ['w_output', 'b_output']:
                fast_weights[key] = weights[key] - FLAGS.update_lr * gradients[key]
            else:
                fast_weights[key] = weights[key]

        # Compute the loss of the network post inner update
        # using an independent set of input/label
        output = self.model.build(inputb, fast_weights, reuse=True)
        task_outputbs.append(output)
        task_lossesb.append(self.loss_func(output, labelb))

        for j in range(num_updates - 1):
            loss = self.loss_func(
                self.model.build(inputa, fast_weights, reuse=True), labela)
            grads = tf.gradients(loss, list(fast_weights.values()))
            if FLAGS.stop_grad:
                grads = [tf.stop_gradient(grad) for grad in grads]

            gradients = dict(zip(fast_weights.keys(), grads))
            for key in weights.keys():
                if key in ['w_output', 'b_output']:
                    fast_weights[key] = fast_weights[key] - FLAGS.update_lr * \
                                        gradients[key]

            output = self.model.build(inputb, fast_weights, reuse=True)
            task_outputbs.append(output)
            task_lossesb.append(self.loss_func(output, labelb))

        # Compute loss/acc using new weights and inputa
        task_outputc = self.model.build(inputa, fast_weights,
                                        reuse=True)
        task_lossc = self.loss_func(task_outputc, labela)
        task_accuracyc = acc_func(task_outputc, labela)

        for task_outputb in task_outputbs:
            task_accuraciesb.append(acc_func(task_outputb, labelb))

        return [task_outputa, task_outputbs, task_outputc,
                task_lossa, task_lossesb, task_lossc,
                task_accuracya, task_accuraciesb, task_accuracyc]

    def _build(self, x, y):
        # a: training data for inner gradient, b: test data for meta gradient

        self.inputa, self.inputb = tf.split(x, 2, axis=1)
        self.labela, self.labelb = tf.split(y, 2, axis=1)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            # Define the weights
            self.weights = self.model.build_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            num_updates = max(self.test_num_updates, FLAGS.num_updates)

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = self.task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # do metalearn for each meta-example in the meta-batch
            # self.inputa has shape (meta_batch_size, batch_size, dim_input)
            # do metalearn on (i, batch_size, dim_input) for i in range(meta_batch_size)
            out_dtype = [tf.float32, [tf.float32] * num_updates, tf.float32,
                         tf.float32, [tf.float32] * num_updates, tf.float32,
                         tf.float32, [tf.float32] * num_updates, tf.float32]
            results = tf.map_fn(
                self.task_metalearn,
                elems=(self.inputa, self.inputb, self.labela, self.labelb),
                dtype=out_dtype,
                parallel_iterations=FLAGS.meta_batch_size
            )

            outputas, outputbs, outputcs = results[:3]
            lossesa, lossesb, lossesc = results[3:6]
            acca, accb, accc = results[6:]

        ## Performance & Optimization
        self.total_loss1 = tf.reduce_mean(lossesa)
        self.total_loss2 = [tf.reduce_mean(l) for l in lossesb]
        self.total_loss3 = tf.reduce_mean(lossesc)
        self.total_acc1 = tf.reduce_mean(acca)
        self.total_acc2 = [tf.reduce_mean(a) for a in accb]
        self.total_acc3 = tf.reduce_mean(accc)
        # after the map_fn
        self.outputas, self.outputbs, self.outputcs = outputas, outputbs, outputcs

        optimizer = tf.train.AdamOptimizer(FLAGS.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(
            self.total_loss2[FLAGS.num_updates-1])
        self.metatrain_op = optimizer.apply_gradients(gvs)

        ## Summaries
        tf.summary.scalar('Pre-update loss', self.total_loss1)
        tf.summary.scalar('Pre-update accuracy', self.total_acc1)
        tf.summary.scalar('Post-update train loss', self.total_loss3)
        tf.summary.scalar('Post-update train accuracy', self.total_acc3)

        for j in range(num_updates):
            tf.summary.scalar(
                'Post-update val loss, step ' + str(j+1), self.total_loss2[j])
            tf.summary.scalar(
                'Post-update val accuracy, step ' + str(j+1), self.total_acc2[j])

from model import _sparse_range, _initializer

class PNKCModel(Model):
    def __init__(self, config):
        self.config = config
        super(PNKCModel, self).__init__(self.config.save_path)

    def build_weights(self):
        n_valence = self.config.n_class_valence
        config = self.config
        weights = {}
        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            if config.sign_constraint_pn2kc:
                if config.initial_pn2kc == 0:
                    if config.sparse_pn2kc:
                        range = _sparse_range(config.kc_inputs)
                    else:
                        range = _sparse_range(config.N_PN)
                else:
                    range = config.initial_pn2kc
                initializer = _initializer(range, config.initializer_pn2kc)
                bias_initializer = tf.constant_initializer(config.kc_bias)
            else:
                initializer = tf.glorot_normal_initializer()
                bias_initializer = tf.zeros_initializer()

            w2 = tf.get_variable(
                'kernel', shape=(config.N_PN, config.N_KC),
                dtype=tf.float32, initializer=initializer)
            if config.sign_constraint_pn2kc:
                w_kc = tf.abs(w2)
            else:
                w_kc = w2
            b_kc = tf.get_variable('bias', shape=(config.N_KC,), dtype=tf.float32,
                                   initializer=bias_initializer)

        with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
            w_output = tf.get_variable(
                'kernel', shape=(config.N_KC, n_valence),
                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            b_output = tf.get_variable(
                'bias', shape=(n_valence,), dtype=tf.float32,
                initializer=tf.zeros_initializer())

        weights['w_kc'] = w_kc
        weights['b_kc'] = b_kc
        weights['w_output'] = w_output
        weights['b_output'] = b_output
        self.weights = weights
        return weights

    def build(self, inp, weights, reuse=False):
        hidden = tf.nn.relu(tf.matmul(inp, weights['w_kc']) + weights['b_kc'])
        if self.config.kc_dropout:
            hidden = tf.layers.dropout(hidden, self.config.kc_dropout_rate, training=True)
        output = tf.matmul(hidden, weights['w_output']) + weights['b_output']

        return output

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
        var_dict = dict()
        # var_dict = {v.name: sess.run(v) for v in tf.trainable_variables()}
        for v in ['w_kc', 'b_kc', 'w_output', 'b_output']:
            var_dict[v] = sess.run(self.weights[v])
        # var_dict['w_glo'] = sess.run(self.weights['w_kc'])
        # var_dict['w_glo'] = sess.run(self.weights['w_kc'])
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)



