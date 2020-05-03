import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import layers
from tensorflow.keras import datasets


class Network:

    def __init__(self):
        with tf.variable_scope('self'):
            self.action = tf.placeholder(tf.int32, shape=[None], name="labels")
            self.reward = tf.placeholder(tf.float32, shape=[None], name="labels")

            self.var = tf.Variable([0, 0, 0, 0, 0], dtype=tf.float32)
            self.sm = tf.nn.softmax(self.var)

        action_onehot = tf.one_hot(self.action, 5, dtype=tf.float32)
        responsible_outputs = tf.reduce_sum(self.sm * action_onehot, 1)
        self.policy_loss = -(tf.log(responsible_outputs)) * self.reward
        self.gradients = tf.gradients(
            self.policy_loss,
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        )

        opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.apply_grads = opt.apply_gradients(zip(self.gradients, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)))


if __name__ == '__main__':
    import numpy as np

    network = Network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        reward = np.array([1, 0, -1, 2, 0])

        for _ in range(1000000):
            p = np.squeeze(sess.run(network.sm))
            act = np.random.choice(5, p=p)
            gradients, policy_loss, _ = sess.run(
                [network.gradients, network.policy_loss, network.apply_grads],
                feed_dict={
                    network.action: [act],
                    network.reward: [reward[act]]
                }
            )
            print(act, policy_loss)
