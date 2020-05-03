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

from Network import Network
from Agent import Agent

if __name__ == '__main__':
    '''
        test run with 1 global and 1 agent
    '''
    import os

    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # physical_devices = tf.config.list_physical_devices('GPU')
    # print("Num GPUs:", len(physical_devices))

    parser = argparse.ArgumentParser(description='Run A3C algorithm on the game Cartpole.')
    parser.add_argument('--algorithm', default='a3c', type=str,
                        help='Choose between \'a3c\' and \'random\'.')
    parser.add_argument('--train', dest='train', action='store_true',
                        help='Train our model.', default=True)
    parser.add_argument('--socket', default=0, type=int,
                        help='Which CPU socket the simulation runs on.')
    parser.add_argument('--num-agents', default=6, type=int,
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

    game = 'PongDeterministic-v4'
    input_shape = [42, 42, 1]
    # input_shape = [42, 42, 4]
    # input_shape = [80, 80, 1]
    action_size = 2

    tf.reset_default_graph()
    _sess = tf.Session()
    # coord = tf.train.Coordinator()
    optimizer = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=.99, momentum=0.0, epsilon=0.1, use_locking=True)
    global_net = Network('global', input_shape, action_size, optimizer)

    thread_list = []
    for id in range(args.num_agents):
        env = gym.make(game)

        single_agent = Agent(
            session=_sess, env=env,  #coord=coord,
            name="Agent_{}".format(id),
            global_network=global_net,
            input_shape=input_shape,
            output_dim=action_size,
            optimizer=optimizer,
        )
        thread_list.append(single_agent)

    _sess.run(tf.global_variables_initializer())
    _sess.run(tf.local_variables_initializer())

    for t in thread_list:
        t.start()

    for t in thread_list:
        t.join()
