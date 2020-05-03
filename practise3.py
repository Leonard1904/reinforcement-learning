import os
import copy
import tensorflow as tf
import numpy as np

from tensorflow import layers
from collections import Iterable


class Network:

    def __init__(self, name, state_size, action_size, opt, feature_layers=None, critic_layers=None, actor_layers=None):
        self.name = name
        self.state_size = state_size
        self.action_size = action_size
        self.optimizer = opt
        self.feature_layers = [
            # layers.Dense(100, activation='relu', name="features"),
            layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.leaky_relu),
            layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.leaky_relu),
            layers.Flatten(),
            layers.Dense(256, activation=tf.nn.leaky_relu, name="features"),
        ] if (feature_layers is None or not isinstance(feature_layers, Iterable)) else feature_layers
        critic_layers = [
            layers.Dense(1, name='value')
        ] if (critic_layers is None or not isinstance(critic_layers, Iterable)) else critic_layers
        # actor_layers = [
        #     layers.Dense(action_size, activation='sigmoid', name='logits')
        # ] if (actor_layers is None or not isinstance(actor_layers, Iterable)) else actor_layers

        self.state = tf.placeholder(tf.float32, shape=[None, *state_size], name="states")
        with tf.variable_scope(self.name):
            self.feature = self._layers_output(self.feature_layers, self.state)
            self.value = self._layers_output(critic_layers, self.feature)
            # self.logits = self._layers_output(actor_layers, self.feature)
            # self.policy = tf.nn.softmax(self.logits, name='policy')
            # self.selected_action = tf.placeholder(tf.float32, [None], name="labels")
            # self.actions_onehot = tf.one_hot(tf.cast(self.selected_action, tf.int32), self.action_size, dtype=tf.float32)
            self.advantages = tf.placeholder(tf.float32, [None])

        if name != 'global':
            # self.value_loss, self.policy_loss, self.entropy_loss, self.total_loss = self._compute_loss()

            # self.responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, 1)
            # responsible_outputs = tf.reduce_sum(self.logits * actions_onehot, 1)
            self.value_loss = (self.advantages - self.value) ** 2

            # self.entropy = -tf.reduce_sum(self.policy * tf.log(self.policy), 1)
            # self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)

            self.total_loss = tf.reduce_mean(0.5 * self.value_loss)  # + self.policy_loss)

            self.gradients = tf.gradients(self.total_loss,
                                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))
            # self.grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
            # self.apply_grads = opt.apply_gradients(
            #     zip(self.grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')))

    @staticmethod
    def _layers_output(layers, x):
        # use deep copy to initialize new objects
        for l in layers:
            x = l.apply(x)
        return x

    @staticmethod
    def transfer_weights(from_scope, to_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

        op_holder = []
        for from_var, to_var in zip(from_vars, to_vars):
            op_holder.append(to_var.assign(from_var))
        return op_holder


if __name__ == '__main__':
    '''
        Example for initializing a network
    '''
    import gym
    from scipy.signal import lfilter
    from scipy.misc import imresize
    import cv2


    def _preprocess(image, height_range=(84, 84)):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (height_range[0], height_range[1]), interpolation=cv2.INTER_LINEAR)

        return image / 255.


    def discount(arr):
        return lfilter([1], [1, -0.99], x=arr[::-1])[::-1]


    tf.reset_default_graph()

    env = gym.make('Pong-v0')
    s = env.reset()
    s = _preprocess(s)

    opt = tf.train.AdamOptimizer(use_locking=True)
    network_ = Network('global', (84, 84, 1), 3, opt)
    network = Network('worker_1', (84, 84, 1), 3, opt)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        value, v_l, loss, gradients = sess.run(
            [network.value, network.value_loss, network.total_loss, network.gradients],
            feed_dict={
                network.state: np.reshape(s, [-1, 84, 84, 1]),
                # network.selected_action: [0],
                network.advantages: [1]
            }
        )
        print(value)
        print(v_l)
        print(loss)
        print(gradients)

        # print(np.squeeze(sess.run(local.test))[-20:])
        # print('-----')
        # input()
        # test training
