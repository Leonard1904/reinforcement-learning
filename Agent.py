import os
import copy
import tensorflow as tf
import numpy as np
import argparse
import scipy.signal
import gym

from tensorflow import layers
from collections import Iterable
import matplotlib.pyplot as plt

import threading
import gym
import time
import cv2

from scipy.misc import imresize
from scipy.signal import lfilter
from Network import Network


class Memory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.rnn_states = []

    def store(self, state, action, reward, value=None, rnn_state=None):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.rnn_states.append(rnn_state)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.rnn_states.clear()

    def size(self):
        return len(self.actions)


class Agent(threading.Thread):
    save_lock = threading.Lock()
    episode = 0
    step = 0
    moving_reward = -21

    def __init__(self, session, env, name, global_network, input_shape, output_dim, optimizer, logdir=None):
        super(Agent, self).__init__()
        self.local = Network(name, input_shape, output_dim, optimizer, logdir)
        self.global_network = global_network

        self.input_shape = input_shape
        self.output_dim = output_dim
        # self.env = gym.make('Pong-v0')
        self.env = env
        self.sess = session
        self.name = name
        self.logdir = logdir
        self.mem = Memory()

    @staticmethod
    def _discounted(rewards, factor):
        return lfilter([1], [1, -factor], x=rewards[::-1])[::-1]

    def _preprocess(self, image, height_range=(35, 193), size=(42, 42)):
        image = image[34:34 + 160, :160]
        image = cv2.resize(image, (80, 80))
        image = cv2.resize(image, (42, 42))
        image = image.mean(2)
        image = image.astype(np.float32)
        image *= (1.0 / 255.0)
        image = np.reshape(image, [42, 42, 1])

        return image

    def play_episode(self):
        s = self.env.reset()
        s = self._preprocess(s)
        rnn_state = self.local.cell_init

        done = False
        ep_reward = 0
        time_step = 0
        step = 0
        pre_action = 0
        while not done:
            if pre_action < 1:
                s, r, done, _ = self.env.step(1)
                s = self._preprocess(s)
                ep_reward += r
                rnn_state = self.local.cell_init
                pre_action += 1
                step += 1
                continue

            a, v, rnn_state_ = self.local.choose_action(self.sess, s, rnn_state)

            s_, r, done, _ = self.env.step(a + 2)
            s_ = self._preprocess(s_)

            ep_reward += r

            self.mem.store(s, a, r, v, rnn_state)

            s = s_
            rnn_state = rnn_state_

            step += 1
            time_step += 1

            if time_step >= 20 or r != 0 or done:
                if r != 0:
                    v_ = 0
                    pre_action = 0
                else:
                    _, v_, _ = self.local.choose_action(self.sess, s, rnn_state)

                _, _, _ = self.train(v_)
                with Agent.save_lock:
                    self.sess.run(self.local.sync)
                self.mem.clear()

                time_step = 0

        Agent.episode += 1
        Agent.step += step
        Agent.moving_reward = Agent.moving_reward * .99 + ep_reward * .01

        print(
            f'{Agent.episode}|{Agent.step:,}|'
            f' Average: {Agent.moving_reward:.2f} '
            f'{self.name} gets {ep_reward} in {step} steps. '
        )

    def run(self):
        with Agent.save_lock:
            self.sess.run(self.local.sync)
        while True:
            self.play_episode()

    def train(self, v_, gamma=0.99, lambda_=1):
        states = np.array(self.mem.states)
        actions = np.array(self.mem.actions)
        values = np.array(self.mem.values + [v_])

        discounted_rewards = self._discounted(self.mem.rewards + [values[-1]], gamma)[:-1]
        rewards = np.array(self.mem.rewards)
        advantages = rewards + (1 - rewards ** 2) * gamma * values[1:] - values[:-1]
        advantages = self._discounted(advantages, gamma * lambda_)

        feed = {
            self.local.states: states,
            self.local.actions: actions,
            self.local.rewards: discounted_rewards,
            self.local.advantage: advantages,
            self.local.cell_in[0]: self.mem.rnn_states[0][0],
            self.local.cell_in[1]: self.mem.rnn_states[0][1],
        }

        rnn_state_, vl, al, _ = self.sess.run([
            self.local.cell_out,
            self.local.value_loss,
            self.local.actor_loss,
            self.local.apply_gradients,
        ], feed)
        return rnn_state_, vl, al
