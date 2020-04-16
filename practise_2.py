import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import layers
from tensorflow.keras import datasets


class Network:

    def __init__(self):
        with tf.variable_scope('self'):
            self.inp = tf.placeholder(tf.float32, shape=[None, 1], name="states")
            self.label = tf.placeholder(tf.float32, shape=[None, 1], name="labels")

            self.var = tf.Variable(0, dtype=tf.float32)
            self.result = self.inp + self.var

        self.loss = self._compute_grad()

    def _compute_grad(self):
        with tf.GradientTape() as tape:
            error = tf.nn.l2_loss(self.label - self.result)
        return tape.gradient(error, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='self'))


if __name__ == '__main__':
    import numpy as np

    network = Network()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        inp = np.array([[1], [3], [3], [5], [6]])

        loss = sess.run(network.loss, feed_dict={network.inp: inp, network.label: inp + 1})
        print(loss)
