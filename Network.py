import os
import copy
import tensorflow as tf
import numpy as np

from tensorflow import layers, initializers
from collections import Iterable


def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer


class Network:

    def __init__(self, name, input_shape, output_dim, optimizer, logdir=None):
        self.output_dim = output_dim
        self.input_shape = input_shape

        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            net = self.states
            for i in range(4):
                net = tf.layers.Conv2D(filters=32,
                                       kernel_size=(3, 3),
                                       strides=2,
                                       activation='elu',
                                       kernel_initializer=initializers.he_uniform(),
                                       bias_initializer=initializers.constant(0.0),
                                       name="conv{}_{}".format(i, name)
                                       )(net)
            net = tf.layers.Flatten()(net)

            net = tf.layers.Dense(256, activation='elu',
                                  kernel_initializer=initializers.random_normal,
                                  bias_initializer=initializers.constant(0.0),
                                  name='dense_{}'.format(name)
                                  )(net)

            # Recurrent network for temporal dependencies
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
            c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
            h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
            self.cell_init = [c_init, h_init]

            c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c])
            h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h])
            self.cell_in = (c_in, h_in)
            rnn_in = tf.expand_dims(net, [0])
            step_size = tf.shape([-1, *input_shape])[:1]
            state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
                lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
                time_major=False)
            lstm_c, lstm_h = lstm_state
            self.cell_out = (lstm_c[:1, :], lstm_h[:1, :])
            # net = tf.reshape(lstm_outputs, [-1, 256])

            # actor network
            logits = tf.layers.Dense(output_dim, kernel_initializer=normalized_columns_initializer(0.01),
                                     name="final_fc_{}".format(name))(net)
            self.action_prob = tf.nn.softmax(logits, name="action_prob")
            # value network
            self.values = tf.squeeze(tf.layers.Dense(1, kernel_initializer=normalized_columns_initializer(1),
                                                     name="values_{}".format(name))(net))

            # actor loss
            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            entropy = -0.01 * tf.reduce_sum(self.action_prob * tf.nn.log_softmax(logits))

            action_onehot = tf.one_hot(self.actions, self.output_dim, name="action_onehot")
            # single_action_prob = tf.reduce_sum(tf.nn.log_softmax(logits) * action_onehot, axis=1)
            # log_action_prob = tf.log(single_action_prob + 1e-7)
            # maximize_objective = log_action_prob * self.advantage
            # self.actor_loss = - tf.reduce_sum(maximize_objective)
            # self.actor_loss = - tf.reduce_sum(
            #     tf.reduce_sum(tf.nn.log_softmax(logits) * action_onehot, [1]) * self.advantage)
            self.actor_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=action_onehot,
                                                                         logits=logits)
            self.actor_loss *= tf.stop_gradient(self.advantage)

            # value loss
            self.value_loss = .5 * tf.reduce_sum(tf.square(self.values - self.rewards))

            self.total_loss = self.actor_loss - entropy + self.value_loss
            self.train_op = optimizer.minimize(self.total_loss)

            self.gradients = tf.gradients(self.total_loss,
                                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name))
            self.grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
            # opt = tf.train.AdamOptimizer(1e-4)
            self.apply_gradients = optimizer.apply_gradients(
                zip(self.grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')))

            self.sync = tf.group(*[v1.assign(v2) for v1, v2 in
                                   zip(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, name),
                                       tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global'))])

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)

    def choose_action(self, sess, state, rnn_state):
        feed = {
            # self.states: [np.transpose(state, (1, 2, 0))],
            self.states: [state],
            self.cell_in[0]: rnn_state[0],
            self.cell_in[1]: rnn_state[1],
        }

        action, value, rnn_state_ = sess.run([self.action_prob, self.values, self.cell_out], feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim), p=action), value, rnn_state_
