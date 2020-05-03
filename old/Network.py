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
            layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.leaky_relu),
            layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.leaky_relu),
            layers.Flatten(),
            layers.Dense(256, activation=tf.nn.leaky_relu, name="features"),
        ] if (feature_layers is None or not isinstance(feature_layers, Iterable)) else feature_layers
        critic_layers = [
            layers.Dense(1, name='value')
        ] if (critic_layers is None or not isinstance(critic_layers, Iterable)) else critic_layers
        actor_layers = [
            layers.Dense(action_size, name='logits')
        ] if (actor_layers is None or not isinstance(actor_layers, Iterable)) else actor_layers

        self.selected_action = tf.placeholder(tf.uint8, [None], name="labels")
        self.actions_onehot = tf.one_hot(self.selected_action, self.action_size, dtype=tf.float32)
        self.advantages = tf.placeholder(tf.float32, [None])
        self.discounted_reward = tf.placeholder(tf.float32, [None])

        self.state = tf.placeholder(tf.float32, shape=[None, *state_size], name="states")
        with tf.variable_scope(self.name):
            self.feature = self._layers_output(self.feature_layers, self.state)
            self.value = self._layers_output(critic_layers, self.feature)
            self.logits = self._layers_output(actor_layers, self.feature)
            self.policy = tf.nn.softmax(self.logits, name='policy')
            # self.value_loss, self.policy_loss, self.entropy, self.total_loss = self._compute_loss()
            # self.target = tf.placeholder(tf.float32, [None])

            responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, 1)
            self.entropy = 0.005 * tf.reduce_sum(-self.policy * tf.log(self.policy + 1e-7), 1)
            self.policy_loss = -tf.reduce_mean((tf.log(responsible_outputs + 1e-7)) * self.advantages + self.entropy)

            self.value_loss = tf.losses.mean_squared_error(self.advantages, tf.squeeze(self.value))

            self.total_loss = 0.5 * self.value_loss + self.policy_loss  # - entropy * 0.005

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)

        # if name != 'global':
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)

        self.gradients_placeholders = []
        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)
        # self.gradients = tf.gradients(self.total_loss,
        #                               tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))
        # self.grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
        # self.apply_grads = opt.apply_gradients(
        #     zip(self.grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')))
        # else:

        self.reward_summary_ph = tf.placeholder(tf.float32, name="reward_summary")
        self.reward_summary = tf.summary.scalar(name='reward_summary', tensor=self.reward_summary_ph)

        self.merged_summary_op = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        self.test = tf.get_default_graph().get_tensor_by_name(os.path.split(self.value.name)[0] + '/kernel:0')

    def summary(self, sess, reward, step):
        summary = sess.run(
            self.merged_summary_op,
            feed_dict={self.reward_summary_ph: reward}
        )
        self.writer.add_summary(summary, step)

    @staticmethod
    def _layers_output(layers, x):
        # use deep copy to initialize new objects
        for l in copy.deepcopy(layers):
            x = l.apply(x)
        return x

    def _compute_loss(self):
        self.selected_action = tf.placeholder(tf.uint8, [None], name="labels")
        self.actions_onehot = tf.one_hot(self.selected_action, self.action_size, dtype=tf.float32)
        self.advantages = tf.placeholder(tf.float32, [None])
        self.discounted_reward = tf.placeholder(tf.float32, [None])
        # self.target = tf.placeholder(tf.float32, [None])

        responsible_outputs = tf.reduce_sum(self.policy * self.actions_onehot, 1)
        entropy = 0.005 * tf.reduce_sum(-self.policy * tf.log(self.policy + 1e-7), 1)
        policy_loss = -tf.reduce_mean((tf.log(responsible_outputs + 1e-7)) * self.advantages + entropy)

        value_loss = tf.losses.mean_squared_error(self.advantages, tf.reshape(self.value, [-1]))

        total_loss = 0.5 * value_loss + policy_loss  # - entropy * 0.005

        return value_loss, policy_loss, entropy, total_loss

    def get_values(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: state})

    def get_action(self, state, sess):
        policy = sess.run(self.policy, feed_dict={self.state: np.reshape(state, (-1, *self.state_size))})

        if np.isnan(np.sum(policy[0])):
            print(policy[0])
            input('policy error')
        return np.random.choice(range(self.action_size), p=policy[0])

    def transfer_weights(self, from_scope):
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)

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

    # arr2 = [0, 0, 0, 0, 0, 1]
    # print(discount(arr2))
    # print((1 - np.array(arr2) ** 2))
    # print(discount(arr2) + (1 - np.array(arr2) ** 2))

    # tf.reset_default_graph()
    #
    # env = gym.make('Pong-v0')
    # s = env.reset()
    # s = _preprocess(s)
    #
    # opt = tf.train.AdamOptimizer(use_locking=True)
    # network_ = Network('global', (84, 84, 1), 3, opt)
    # network = Network('worker_1', (84, 84, 1), 3, opt)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #
    #     grad = sess.run(
    #         network.grads,
    #         feed_dict={
    #             network.state: np.reshape(s, [-1, 84, 84, 1]),
    #             network.selected_action: [0],
    #             network.advantages: [1],
    #             network.discounted_reward: [0.5],
    #         }
    #     )
    #     print(grad)

        # print(np.squeeze(sess.run(local.test))[-20:])
        # print('-----')
        # input()
        # test training
