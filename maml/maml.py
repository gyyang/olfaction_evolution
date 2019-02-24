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
    return tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)


class MAML:
    def __init__(self, config):
        """MAML model."""
        self.model = Model(config)
        self.loss_func = xent

        self._build()

    def _build(self):
        # a: training data for inner gradient, b: test data for meta gradient
        self.input = tf.placeholder(tf.float32) # (meta_batch_size, batch_size, dim_inputs)
        self.label = tf.placeholder(tf.int32) # (meta_batch_size, batch_size, dim_outputs)

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
        self.dim_input = config.N_ORN
        self.dim_output = config.N_CLASS
        self.dim_hidden = [40, 40]

    def build_weights(self):
        n_layer = len(self.dim_hidden)
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,n_layer):
            weights['w'+str(i+1)] = tf.Variable(
                tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(n_layer+1)] = tf.Variable(
            tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(n_layer+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def build(self, inp, weights, reuse=False):
        n_layer = len(self.dim_hidden)
        hidden = tf.matmul(inp, weights['w1']) + weights['b1']
        hidden = normalize(hidden, activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1, n_layer):
            hidden = tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)]
            hidden = normalize(hidden, activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        hidden = tf.matmul(hidden, weights['w'+str(n_layer+1)]) + weights['b'+str(n_layer+1)]
        return hidden



