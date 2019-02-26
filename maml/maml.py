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


class MAML:
    def __init__(self, x, y, config):
        """MAML model."""
        self.model = PNKCModel(config)
        self.loss_func = xent
        # self.loss_func = mse
        self.save_pickle = self.model.save_pickle

        self._build(x, y)

    def _build(self, x, y):
        # a: training data for inner gradient, b: test data for meta gradient

        self.inputa, self.inputb = tf.split(x, 2, axis=1)
        self.labela, self.labelb = tf.split(y, 2, axis=1)

        with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as training_scope:
            # Define the weights
            self.weights = weights = self.model.build_weights()

            def task_metalearn(inp, reuse=True):
                """ Perform gradient descent for one task in the meta-batch.

                Args:
                    inp: a sequence unpacked to inputa, inputb, labela, labelb
                    inputa: tensor (batch_size, dim_input)

                Returns:
                    task_output: a sequence unpacked to outputa, outputb, lossa, lossb
                """
                inputa, inputb, labela, labelb = inp

                task_outputa = self.model.build(inputa, weights, reuse=reuse)  # only reuse on the first iter
                task_lossa = self.loss_func(task_outputa, labela)
                # Compute accuracy
                task_accuracya = tf.reduce_mean(tf.to_float(tf.equal(
                    tf.argmax(task_outputa, 1), tf.argmax(labela, 1))))

                grads = tf.gradients(task_lossa, list(weights.values()))
                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]

                # manually construct the weights post inner gradient descent
                # Notice that this doesn't have to be through gradient descent
                gradients = dict(zip(weights.keys(), grads))
                fast_weights = dict()
                for key in weights.keys():
                    fast_weights[key] = weights[key]-FLAGS.update_lr*gradients[key]

                # Compute the loss of the network post inner update
                # using an independent set of input/label
                task_outputb = self.model.build(inputb, fast_weights, reuse=True)
                task_lossb = self.loss_func(task_outputb, labelb)
                task_accuracyb = tf.reduce_mean(tf.to_float(tf.equal(
                    tf.argmax(task_outputb, 1), tf.argmax(labelb, 1))))

                # Compute loss/acc using new weights and inputa
                task_outputc = self.model.build(inputa, fast_weights,
                                                reuse=True)
                task_lossc = self.loss_func(task_outputc, labela)
                task_accuracyc = tf.reduce_mean(tf.to_float(tf.equal(
                    tf.argmax(task_outputc, 1), tf.argmax(labela, 1))))

                return [task_outputa, task_outputb, task_outputc,
                        task_lossa, task_lossb, task_lossc,
                        task_accuracya, task_accuracyb, task_accuracyc]

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # do metalearn for each meta-example in the meta-batch
            # self.inputa has shape (meta_batch_size, batch_size, dim_input)
            # do metalearn on (i, batch_size, dim_input) for i in range(meta_batch_size)
            results = tf.map_fn(
                task_metalearn,
                elems=(self.inputa, self.inputb, self.labela, self.labelb),
                dtype=[tf.float32]*9,
                parallel_iterations=FLAGS.meta_batch_size
            )

            outputas, outputbs, outputcs = results[:3]
            lossesa, lossesb, lossesc = results[3:6]
            acca, accb, accc = results[6:]

        ## Performance & Optimization
        self.total_loss1 = tf.reduce_mean(lossesa)
        self.total_loss2 = tf.reduce_mean(lossesb)
        self.total_loss3 = tf.reduce_mean(lossesc)
        self.total_acc1 = tf.reduce_mean(acca)
        self.total_acc2 = tf.reduce_mean(accb)
        self.total_acc3 = tf.reduce_mean(accc)
        # after the map_fn
        self.outputas, self.outputbs, self.outputcs = outputas, outputbs, outputcs

        optimizer = tf.train.AdamOptimizer(FLAGS.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_loss2)
        self.metatrain_op = optimizer.apply_gradients(gvs)


class PNKCModel(Model):
    def __init__(self, config):
        self.config = config
        super(PNKCModel, self).__init__(self.config.save_path)

    def build_weights(self):
        n_valence = self.config.n_class_valence
        config = self.config
        weights = {}
        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            w2 = tf.get_variable(
                'kernel', shape=(config.N_ORN, config.N_KC),
                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            if config.sign_constraint_pn2kc:
                w_kc = tf.abs(w2)
            else:
                w_kc = w2
            b_kc = tf.get_variable('bias', shape=(config.N_KC,), dtype=tf.float32,
                                   initializer=tf.zeros_initializer())

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
        output = tf.nn.relu(tf.matmul(hidden, weights['w_output']) +
                            weights['b_output'])
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



