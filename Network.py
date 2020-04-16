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
        actor_layers = [
            layers.Dense(action_size, name='logits')
        ] if (actor_layers is None or not isinstance(actor_layers, Iterable)) else actor_layers

        # with tf.device("/cpu:{}".format(idx)):
        self.state = tf.placeholder(tf.float32, shape=[None, *state_size], name="states")
        with tf.variable_scope(self.name):
            self.feature = self._layers_output(self.feature_layers, self.state)
            # self.feature_ = self._layers_output(self.feature_layers, self.state)
            self.value = self._layers_output(critic_layers, self.feature)
            self.logits = self._layers_output(actor_layers, self.feature)
            self.policy = tf.nn.softmax(self.logits, name='policy')

        if name != 'global':
            self.value_loss, self.policy_loss, self.entropy_loss, self.total_loss = self._compute_loss()
            self.gradients = tf.gradients(self.total_loss,
                                          tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name))
            grads, self.grad_norms = tf.clip_by_global_norm(self.gradients, 40.0)
            self.apply_grads = opt.apply_gradients(
                zip(grads, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')))
        else:
            # self.writer = tf.summary.FileWriter('./graphs', sess.graph)

            self.reward_summary_ph = tf.placeholder(tf.float32, name="reward_summary")
            self.reward_summary = tf.summary.scalar(name='reward_summary', tensor=self.reward_summary_ph)

            self.merged_summary_op = tf.summary.merge_all()
            self.writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        # self.test = tf.get_default_graph().get_tensor_by_name(os.path.split(self.value.name)[0] + '/kernel:0')

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
        self.selected_action = tf.placeholder(tf.int32, [None], name="labels")
        actions_onehot = tf.one_hot(self.selected_action, self.action_size, dtype=tf.float32)
        self.discounted_reward = tf.placeholder(tf.float32, [None])
        self.advantages = tf.placeholder(tf.float32, [None])

        responsible_outputs = tf.reduce_sum(self.policy * actions_onehot, [1])
        value_loss = 0.5 * tf.reduce_sum(tf.square(self.discounted_reward - tf.reshape(self.value, [-1])))

        entropy = - tf.reduce_sum(self.policy * tf.log(self.policy + 10e-13))
        # entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.policy, logits=self.logits)
        policy_loss = -tf.reduce_sum(tf.log(responsible_outputs + 10e-13) * self.advantages)

        total_loss = 0.5 * value_loss + policy_loss - entropy * 0.05

        return value_loss, policy_loss, entropy, total_loss

    def get_value(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: np.reshape(state, [-1, *self.state_size])})[0][0]

    def get_values(self, state, sess):
        return sess.run(self.value, feed_dict={self.state: np.reshape(state, [-1, *self.state_size])})

    def get_action(self, state, sess):
        policy = sess.run(self.policy, feed_dict={self.state: np.reshape(state, [-1, *self.state_size])})
        return np.random.choice(range(self.action_size), p=policy[0])

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


    def _preprocess(image, height_range=(35, 193), bg=(144, 72, 17)):
        image = image[height_range[0]:height_range[1], ...]
        image = imresize(image, (80, 80), interp="nearest")

        H, W, _ = image.shape

        R = image[..., 0]
        G = image[..., 1]
        B = image[..., 2]

        cond = (R == bg[0]) & (G == bg[1]) & (B == bg[2])

        image = np.zeros((H, W))
        image[~cond] = 1

        image = np.expand_dims(image, axis=2)

        return image


    def discount(arr):
        return lfilter([1], [1, -0.99], x=arr[::-1])[::-1]


    tf.reset_default_graph()

    env = gym.make('Pong-v0')

    opt = tf.train.AdamOptimizer(use_locking=True)
    network = Network('global', (80, 80, 1), 3, opt)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        a = network.get_action(_preprocess(env.reset()), sess)
        value = network.get_value(_preprocess(env.reset()), sess)
        print(a)
        print(value)

        arr = np.array([0, 0, 0, 0])
        print(discount(arr))

        arr = np.array([0, 0, 0, 0])
        print(arr)
        print(arr.tolist() + [1])
        print(discount(arr.tolist() + [1])[:-1])

        # print(np.squeeze(sess.run(local.test))[-20:])
        # print('-----')
        # input()
        # test training
