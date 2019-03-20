""" Code for the MAML algorithm and network definitions.

Adapted from Chelsea Finn's code
"""
from __future__ import print_function

import os
import pickle

import tensorflow as tf
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.python.platform import flags

from model import Model, FullModel

# FLAGS = flags.FLAGS

# def normalize(inp, activation, reuse, scope):
#     if FLAGS.meta_norm == 'batch_norm':
#         return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#     elif FLAGS.meta_norm.norm == 'layer_norm':
#         return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
#     elif FLAGS.meta_norm.norm == 'None':
#         if activation is not None:
#             return activation(inp)
#         else:
#             return inp

## Loss functions
def mse(pred, label):
    pred = tf.reshape(pred, [-1]) ## Hi peter I miss you <3 come back home
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

        self.model = FullModel(x=None, y=None, config=config, meta_learn=True)
        self.loss_func = lambda logits, y: self.model.loss_func(logits, None, y)
        self.acc_func = lambda logits, y: self.model.accuracy_func(logits, None, y)

        # self.model = PNKCModel(config)
        # self.loss_func = xent
        # self.acc_func = acc_func

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
        def _update_weights(loss, lr_dict, weights):
            '''
            Take gradients WRT trainable weights, and updates weights based on these gradients.
            Weights that are not trainable are not updated.

            :param loss: loss to take gradients to
            :param lr_dict: learning rates associated with weights to update
            :param weights:
            :return:
            '''
            grads = tf.gradients(loss, list(weights.values()))
            if self.model.config.meta_stop_grad:
                grads = [tf.stop_gradient(grad) for grad in grads]
            gradients = dict(zip(weights.keys(), grads))
            # manually construct the weights post inner gradient descent
            # Notice that this doesn't have to be through gradient descent
            new_weights = dict()
            for key in weights.keys():
                if key in lr_dict.keys():
                    new_weights[key] = weights[key] - lr_dict[key] * gradients[key]
                else:
                    new_weights[key] = weights[key]
            return new_weights

        weights = self.weights
        num_updates = max(self.test_num_updates, self.model.config.meta_num_updates)
        inputa, inputb, labela, labelb = inp
        task_outputbs, task_lossesb, task_accuraciesb = [], [], []

        task_outputa = self.model.build_activity(inputa, weights, training=True, reuse=reuse)[0]  # only reuse on the first iter
        task_lossa = self.loss_func(task_outputa, labela)
        task_accuracya = self.acc_func(task_outputa, labela)

        lr_dict = {
            # 'w_orn': 0,
            # 'b_orn': 0,
            # 'w_glo': 0,
            # 'b_glo': 0,
            'w_output': tf.math.minimum(1.0, self.update_lr[0]),
            # 'b_output': self.update_lr[1]
                          }
        fast_weights = _update_weights(task_lossa, lr_dict, weights)

        # Compute the loss of the network post inner update
        # using an independent set of input/label
        output = self.model.build_activity(inputb, fast_weights, training=True, reuse=True)[0]
        task_outputbs.append(output)
        task_lossesb.append(self.loss_func(output, labelb))

        for j in range(num_updates - 1):
            loss = self.loss_func(
                self.model.build_activity(inputa, fast_weights, training=True, reuse=True)[0], labela)
            fast_weights = _update_weights(loss, lr_dict, fast_weights)
            output = self.model.build_activity(inputb, fast_weights, training=True, reuse=True)[0]
            task_outputbs.append(output)
            task_lossesb.append(self.loss_func(output, labelb))

        # Compute loss/acc using new weights and inputa
        task_outputc = self.model.build_activity(inputa, fast_weights,
                                                 training=True, reuse=True)[0]
        task_lossc = self.loss_func(task_outputc, labela)
        task_accuracyc = self.acc_func(task_outputc, labela)

        for task_outputb in task_outputbs:
            task_accuraciesb.append(self.acc_func(task_outputb, labelb))

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
            self.update_lr = tf.get_variable('lr', shape=(2), dtype=tf.float32,
                                             initializer=tf.constant_initializer([.1, -.01]))

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            num_updates = max(self.test_num_updates, self.model.config.meta_num_updates)

            if self.model.config.meta_norm is not 'None':
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
                parallel_iterations=self.model.config.meta_batch_size
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

        excludes = list()
        if not self.model.config.train_orn2pn:
            excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='model/layer1')
        if not self.model.config.train_pn2kc:
            excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='model/layer2/kernel:0')
        if not self.model.config.train_kc_bias:
            excludes += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='model/layer2/bias:0')
        excludes += [self.update_lr]
        var_list = [v for v in tf.trainable_variables() if v not in excludes]
        print('Training variables')
        for v in var_list:
            print(v)
        optimizer = tf.train.AdamOptimizer(self.model.config.meta_lr)
        self.gvs = gvs = optimizer.compute_gradients(self.total_loss2[self.model.config.meta_num_updates-1], var_list)
        self.metatrain_op = optimizer.apply_gradients(gvs)

        training_learning_rate = True
        update_lr_learning_rate = .01
        if training_learning_rate:
            print(self.update_lr)
            optimizer_lr = tf.train.AdamOptimizer(update_lr_learning_rate)
            self.gvs_lr = gvs = optimizer_lr.compute_gradients(self.total_loss2[self.model.config.meta_num_updates - 1], self.update_lr)
            self.metatrain_op_lr = optimizer_lr.apply_gradients(gvs)
        else:
            self.metatrain_op_lr = None

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
        n_class = self.config.N_CLASS
        config = self.config
        weights = {}

        with tf.variable_scope('layer1', reuse=tf.AUTO_REUSE):
            if config.sign_constraint_orn2pn:
                range = _sparse_range(config.N_ORN)
                initializer = _initializer(range, config.initializer_orn2pn)
                bias_initializer = tf.glorot_normal_initializer
            else:
                initializer = tf.glorot_normal_initializer
                bias_initializer = tf.glorot_normal_initializer

            w_orn = tf.get_variable('kernel', shape=(config.N_ORN, config.N_PN),
                                    dtype=tf.float32,
                                    initializer=initializer)

            b_orn = tf.get_variable('bias', shape=(config.N_PN,), dtype=tf.float32,
                                    initializer=bias_initializer)
            if config.sign_constraint_orn2pn:
                w_orn = tf.abs(w_orn)

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
                w_glo = tf.abs(w2)
            else:
                w_glo = w2
            b_glo = tf.get_variable('bias', shape=(config.N_KC,), dtype=tf.float32,
                                   initializer=bias_initializer)

        with tf.variable_scope('layer3', reuse=tf.AUTO_REUSE):
            w_output = tf.get_variable(
                'kernel', shape=(config.N_KC, n_class),
                dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            b_output = tf.get_variable(
                'bias', shape=(n_class,), dtype=tf.float32,
                initializer=tf.zeros_initializer())

        weights['w_orn'] = w_orn
        weights['b_orn'] = b_orn
        weights['w_glo'] = w_glo
        weights['b_glo'] = b_glo
        weights['w_output'] = w_output
        weights['b_output'] = b_output
        self.weights = weights
        return weights

    def build_activity(self, inp, weights, training=True, reuse=False):
        # pn = tf.nn.relu(tf.matmul(inp, weights['w_orn']) + weights['b_orn'])
        # kc = tf.nn.relu(tf.matmul(pn, weights['w_glo']) + weights['b_glo'])
        kc = tf.nn.relu(tf.matmul(inp, weights['w_glo']) + weights['b_glo'])
        if self.config.kc_dropout:
            kc = tf.layers.dropout(kc, self.config.kc_dropout_rate, training= training)
        logits = tf.matmul(kc, weights['w_output']) + weights['b_output']
        logits2 = None
        return logits, logits

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
        for k in self.weights.keys():
            var_dict[k] = sess.run(self.weights[k])
        # var_dict['w_glo'] = sess.run(self.weights['w_glo'])
        # var_dict['w_glo'] = sess.run(self.weights['w_glo'])
        with open(fname, 'wb') as f:
            pickle.dump(var_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Model weights saved in path: %s" % save_path)



