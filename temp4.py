import os
import copy
import tensorflow as tf
import numpy as np
import argparse

from tensorflow import layers
from collections import Iterable

import threading
import gym
import time
import cv2

from scipy.misc import imresize
from scipy.signal import lfilter


class Network:

    def __init__(self, name, input_shape, output_dim, logdir=None):
        """Network structure is defined here
        Args:
            name (str): The name of scope
            input_shape (list): The shape of input image [H, W, C]
            output_dim (int): Number of actions
            logdir (str, optional): directory to save summaries
                TODO: create a summary op
        """
        self.output_dim = output_dim
        self.input_shape = input_shape

        with tf.variable_scope(name):
            self.states = tf.placeholder(tf.float32, shape=[None, *input_shape], name="states")
            net = self.states
            net = tf.layers.Conv2D(filters=16,
                                   kernel_size=(8, 8),
                                   strides=(4, 4),
                                   activation='relu',
                                   name="conv_{}".format(name)
                                   )(net)
            net = tf.layers.Conv2D(filters=32,
                                   kernel_size=(4, 4),
                                   strides=(2, 2),
                                   activation='relu',
                                   name="conv_{}".format(name)
                                   )(net)

            net = tf.layers.Flatten()(net)
            self.net = tf.layers.Dense(256, activation='relu',
                                       name='dense_{}'.format(name)
                                       )(net)

            # actor network
            actions = tf.layers.Dense(output_dim, name="final_fc_{}".format(name))(self.net)
            self.action_prob = tf.nn.softmax(actions, name="action_prob")

            self.actions = tf.placeholder(tf.uint8, shape=[None], name="actions")
            self.rewards = tf.placeholder(tf.float32, shape=[None], name="rewards")
            self.advantage = tf.placeholder(tf.float32, shape=[None], name="advantage")

            action_onehot = tf.one_hot(self.actions, self.output_dim, name="action_onehot")
            single_action_prob = tf.reduce_sum(self.action_prob * action_onehot, axis=1)

            entropy = - self.action_prob * tf.log(self.action_prob + 1e-7)
            entropy = tf.reduce_sum(entropy, axis=1)

            log_action_prob = tf.log(single_action_prob + 1e-7)
            maximize_objective = log_action_prob * self.advantage + entropy * 0.005
            actor_loss = - tf.reduce_mean(maximize_objective)

            # value network
            self.values = tf.squeeze(tf.layers.Dense(1, name="values_{}".format(name))(self.net))
            value_loss = tf.losses.mean_squared_error(labels=self.rewards, predictions=self.values)
            self.total_loss = actor_loss + value_loss * .5

            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.01, decay=.99)

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        self.gradients = self.optimizer.compute_gradients(self.total_loss, var_list)
        self.gradients_placeholders = []

        for grad, var in self.gradients:
            self.gradients_placeholders.append((tf.placeholder(var.dtype, shape=var.get_shape()), var))
        self.apply_gradients = self.optimizer.apply_gradients(self.gradients_placeholders)

        if logdir:
            loss_summary = tf.summary.scalar("total_loss", self.total_loss)
            value_summary = tf.summary.histogram("values", self.values)

            self.summary_op = tf.summary.merge([loss_summary, value_summary])
            self.summary_writer = tf.summary.FileWriter(logdir)

    def _layers_output(self, layers, x):
        # use deep copy to initialize new objects
        for l in copy.deepcopy(layers):
            x = l.apply(x)
        return x

    def choose_action(self, sess, states):
        """
        Args:
            states (2-D array): (N, H, W, 1)
        """
        states = np.reshape(states, [-1, *self.input_shape])
        feed = {
            self.states: states
        }

        action = sess.run(self.action_prob, feed)
        action = np.squeeze(action)

        return np.random.choice(np.arange(self.output_dim), p=action)


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def store(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()

    def size(self):
        return len(self.actions)


def copy_src_to_dst(from_scope, to_scope):
    """Creates a copy variable weights operation
    Args:
        from_scope (str): The name of scope to copy from
            It should be "global"
        to_scope (str): The name of scope to copy to
            It should be "thread-{}"
    Returns:
        list: Each element is a copy operation
    """
    from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, from_scope)
    to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, to_scope)

    op_holder = []
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


class Agent(threading.Thread):

    global_episode = 0

    def __init__(self, session, env, coord, name, global_network, input_shape, output_dim, logdir=None):
        """Agent worker thread
        Args:
            session (tf.Session): Tensorflow session needs to be shared
            env (gym.env): Gym environment
            coord (tf.train.Coordinator): Tensorflow Queue Coordinator
            name (str): Name of this worker
            global_network (A3CNetwork): Global network that needs to be updated
            input_shape (list): Required for local A3CNetwork (H, W, C)
            output_dim (int): Number of actions
            logdir (str, optional): If logdir is given, will write summary
                TODO: Add summary
        """
        super(Agent, self).__init__()
        self.local = Network(name, input_shape, output_dim, logdir)
        self.global_to_local = copy_src_to_dst("global", name)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        self.env = env
        self.sess = session
        self.coord = coord
        self.name = name
        self.logdir = logdir


    def _discounted_reward(self, rewards):
        discounted_r = np.zeros_like(rewards, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(rewards))):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * 0.99 + rewards[t]
            discounted_r[t] = running_add

        return discounted_r
        # return lfilter([1], [1, -0.99], x=rewards[::-1])[::-1]

    def _preprocess(self, image, height_range=(35, 193), bg=(144, 72, 17)):
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

    def print(self, episode, reward, step):
        message = "Agent({}: name={}, reward={}, step={})".format(episode, self.name, reward, step)
        print(message)

    def play_episode(self):
        self.sess.run(self.global_to_local)

        states = []
        actions = []
        rewards = []

        s = self.env.reset()
        s = self._preprocess(s)
        state_diff = s

        done = False
        total_reward = 0
        time_step = 0
        step = 0
        while not done:
            a = self.local.choose_action(self.sess, state_diff)
            s2, r, done, _ = self.env.step(a + 1)
            s2 = self._preprocess(s2)

            total_reward += r

            states.append(state_diff)
            actions.append(a)
            rewards.append(r)

            state_diff = s2 - s
            s = s2
            step += 1

            if r == -1 or r == 1 or done:
                time_step += 1
                if time_step >= 5 or done:
                    self.train(states, actions, rewards)
                    self.sess.run(self.global_to_local)
                    states, actions, rewards = [], [], []
                    time_step = 0

        Agent.global_episode += 1

        self.print(Agent.global_episode, total_reward, step)

    def run(self):
        while not self.coord.should_stop():
            self.play_episode()

    def train(self, states, actions, rewards):
        states = np.array(states)
        actions = np.array(actions)
        rewards = np.array(rewards)

        feed = {
            self.local.states: states
        }

        values = self.sess.run(self.local.values, feed)

        rewards = self._discounted_reward(rewards)
        rewards -= np.mean(rewards)
        rewards /= np.std(rewards)

        advantage = rewards - values
        advantage -= np.mean(advantage)
        advantage /= np.std(advantage) + 1e-8

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: rewards,
            self.local.advantage: advantage
        }

        gradients = self.sess.run(self.local.gradients, feed)

        feed = []
        for (grad, _), (placeholder, _) in zip(gradients, self.global_network.gradients_placeholders):
            feed.append((placeholder, grad))

        feed = dict(feed)
        self.sess.run(self.global_network.apply_gradients, feed)


if __name__ == '__main__':
    '''
        test run with 1 global and 1 agent
    '''
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))

    parser = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
    parser.add_argument('--algorithm', default='a3c', type=str,
                        help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train our model.', default=True)
    parser.add_argument('--socket', default=0, type=int,
                        help='Which CPU socket the simulation runs on.')
    parser.add_argument('--num-agents', default=8, type=int,
                        help='Number of agent.')
    parser.add_argument('--lr', default=0.001,
                        help='Learning rate for the shared optimizer.')
    parser.add_argument('--update-freq', default=5, type=int,
                        help='How often to update the global model.')
    parser.add_argument('--max-steps', default=500 * 1e+6, type=int,
                        help='Global maximum number of steps to run.')
    parser.add_argument('--gamma', default=0.99,
                        help='Discount factor of rewards.')
    parser.add_argument('--save-dir', default='./tmp/', type=str,
                        help='Directory in which you desire to save the model.')
    args = parser.parse_args()

    game = 'Pong-v0'
    # state_size = (84, 84, 4)
    input_shape = [80, 80, 1]
    action_size = 3

    tf.reset_default_graph()
    _sess = tf.Session()
    coord = tf.train.Coordinator()
    global_net = Network('global', input_shape, action_size)

    thread_list = []
    for id in range(args.num_agents):
        env = gym.make(game)

        single_agent = Agent(
            session=_sess, env=env, coord=coord,
            name="thread_{}".format(id),
            global_network=global_net,
            input_shape=input_shape,
            output_dim=action_size,
        )
        thread_list.append(single_agent)

    # agents = [
    #     Agent('worker_{}'.format(i),
    #           game, state_size, action_size,
    #           global_net, _sess,
    #           args,
    #           feature_layers)
    #     for i in range(args.num_agents)
    # ]
    _sess.run(tf.global_variables_initializer())
    _sess.run(tf.local_variables_initializer())

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
