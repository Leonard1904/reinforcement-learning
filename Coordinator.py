import gym
import argparse
import tensorflow as tf

from tensorflow import layers
from Network import Network
from Agent import Agent

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
    parser.add_argument('--num-agents', default=16, type=int,
                        help='Number of agent.')
    parser.add_argument('--lr', default=0.001,
                        help='Learning rate for the shared optimizer.')
    parser.add_argument('--update-freq', default=20, type=int,
                        help='How often to update the global model.')
    parser.add_argument('--max-steps', default=500 * 1e+6, type=int,
                        help='Global maximum number of steps to run.')
    parser.add_argument('--gamma', default=0.99,
                        help='Discount factor of rewards.')
    parser.add_argument('--save-dir', default='./tmp/', type=str,
                        help='Directory in which you desire to save the model.')
    args = parser.parse_args()

    game = 'Pong-v0'
    env = gym.make(game)
    state_size = (80, 80, 1)
    action_size = 3


    def act_post_func(act):
        return act + 1


    # feature_layers = [
    #     layers.Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), activation=tf.nn.leaky_relu),
    #     layers.Conv2D(filters=32, kernel_size=(4, 4), strides=(2, 2), activation=tf.nn.leaky_relu),
    #     layers.Flatten(),
    #     layers.Dense(256, activation=tf.nn.leaky_relu, name="features"),
    # ]
    feature_layers = None

    config = tf.ConfigProto(
        intra_op_parallelism_threads=5,
        inter_op_parallelism_threads=10,
        allow_soft_placement=True,
        device_count={'CPU': 20}
    )
    _sess = tf.Session(config=config)
    opt = tf.train.RMSPropOptimizer(learning_rate=7e-4, decay=0.99, epsilon=0.1)
    # opt = tf.train.AdamOptimizer(learning_rate=args.lr, use_locking=True)
    global_net = Network('global', state_size, action_size, opt, feature_layers)

    agents = [
        Agent('worker_{}'.format(i),
              game, state_size, action_size,
              global_net, _sess,
              args,
              feature_layers)
        for i in range(args.num_agents)
    ]
    _sess.run(tf.global_variables_initializer())
    _sess.run(tf.local_variables_initializer())
    for i, agent in enumerate(agents):
        print("Starting worker {}".format(i))
        agent.start()
    [w.join() for w in agents]
