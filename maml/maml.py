""" Code for the MAML algorithm and network definitions.

Adpated from Chelsea Finn's code
"""
from __future__ import print_function
import tensorflow as tf

from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

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
    def __init__(self, config):
        """MAML model."""
        self.model = Model(config)
        self.loss_func = xent

        self._build()

    def _build(self):
        # a: training data for inner gradient, b: test data for meta gradient
        self.input = tf.placeholder(tf.float32) # (meta_batch_size, batch_size, dim_inputs)
        self.label = tf.placeholder(tf.float32) # (meta_batch_size, batch_size, dim_outputs)

        self.inputa, self.inputb = tf.split(self.input, 2, axis=1)
        self.labela, self.labelb = tf.split(self.label, 2, axis=1)

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

                return [task_outputa, task_outputb, task_lossa, task_lossb]

            if FLAGS.norm is not 'None':
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice.
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), False)

            # do metalearn for each meta-example in the meta-batch
            # self.inputa has shape (meta_batch_size, batch_size, dim_input)
            # do metalearn on (i, batch_size, dim_input) for i in range(meta_batch_size)
            outputas, outputbs, lossesa, lossesb = tf.map_fn(
                task_metalearn,
                elems=(self.inputa, self.inputb, self.labela, self.labelb),
                dtype=[tf.float32, tf.float32, tf.float32, tf.float32],
                parallel_iterations=FLAGS.meta_batch_size
            )

        ## Performance & Optimization
        self.total_loss1 = tf.reduce_mean(lossesa)
        self.total_loss2 = tf.reduce_mean(lossesb)
        # after the map_fn
        self.outputas, self.outputbs = outputas, outputbs

        optimizer = tf.train.AdamOptimizer(FLAGS.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_loss2)
        self.metatrain_op = optimizer.apply_gradients(gvs)


class Model():
    def __init__(self, config):
        self.config = config

    def build_weights(self):
        n_valence = 3  # TODO: fix this
        config = self.config
        weights = {}
        with tf.variable_scope('layer2', reuse=tf.AUTO_REUSE):
            w2 = tf.get_variable(
                'kernel', shape=(config.N_ORN, config.N_KC),
                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            w_kc = tf.abs(w2)
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
        return weights

    def build(self, inp, weights, reuse=False):
        hidden = tf.nn.relu(tf.matmul(inp, weights['w_kc']) + weights['b_kc'])
        output = tf.nn.relu(tf.matmul(hidden, weights['w_output']) +
                            weights['b_output'])
        return output



