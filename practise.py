import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import layers
from tensorflow.keras import datasets


class Network:

    def __init__(self, state_size, action_size):
        self.states = tf.placeholder(tf.float32, shape=[None, *state_size], name="states")
        self.labels = tf.placeholder(tf.int32, shape=[None, 1], name="labels")  # size 1 because sparse loss is used.

        # conv1 = layers.conv2d(self.states, filters=16, kernel_size=(8, 8), strides=(4, 4), activation='relu',
        #                       name="conv1"),
        # conv2 = layers.conv2d(conv1, filters=32, kernel_size=(4, 4), strides=(2, 2), activation='relu', name="conv2"),
        # flatten = layers.flatten(conv2),
        # dense = layers.dense(flatten, 256, activation='relu', name="features"),
        #
        # self.logits = layers.dense(dense, action_size, name="logits")
        # self.value = layers.dense(dense, 1, name="values")

        # conv1 = conv2d(self.states, filters=32, kernel_size=(3, 3), name='conv1')

        with tf.variable_scope('layers'):
            conv1 = layers.conv2d(self.states, filters=32, kernel_size=(3, 3), activation='relu', name='conv1')
            conv2 = layers.conv2d(conv1, filters=64, kernel_size=(3, 3), activation='relu', name='conv2')
            max_pool = layers.max_pooling2d(conv2, 2, 1, name='max_pool')
            drop_1 = layers.dropout(max_pool, 0.25, name='drop1')
            flatten = layers.flatten(drop_1, name='flatten')
            dense = layers.dense(flatten, 128, activation='relu', name='dense')
            drop2 = layers.dropout(dense, 0.5, name='drop2')
            logits = layers.dense(drop2, action_size, activation='softmax', name='logits')
            self.output = tf.nn.softmax(logits, name='output')
        # tf.one_hot(tf.arg_max(self.output, 1), depth=10)
        print(tf.arg_max(self.output, 1))
        self.test = tf.one_hot(tf.arg_max(self.output, 1), depth=10)
        # input()
        self.cost = tf.losses.sparse_softmax_cross_entropy(self.labels, logits)
        self.acc, self.acc_op = tf.metrics.accuracy(self.labels, tf.arg_max(self.output, 1))

        # self.grad = tf.gradients(self.cost, self.states, stop_gradients=[self.states])
        self.grad = tf.gradients(self.cost, tf.trainable_variables())

        self.optimizer = tf.train.AdamOptimizer()
        # print(tf.trainable_variables())
        # print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layers'))
        # print(self.grad)
        self.apply_grad = self.optimizer.apply_gradients(
            zip(self.grad, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='layers'))
        )


if __name__ == '__main__':
    import numpy as np

    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_images, test_images = train_images[..., np.newaxis], test_images[..., np.newaxis]

    tf.reset_default_graph()
    network = Network((28, 28, 1), 10)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # print(train_labels[:2].shape)
        # output = sess.run(network.test, feed_dict={network.states: train_images[:2, :, :, :]})
        # print(output)
        # print(train_labels[:2][:, None])
        # grad = sess.run(network.grad,
        #                 feed_dict={
        #                     network.states: train_images[:2, :, :, :],
        #                     network.labels: train_labels[:2][:, None],
        #                 })
        # for g in grad:
        #     print(g.shape)

        idx = np.random.choice(range(train_images.shape[0]), 64, replace=False)
        acc, acc_op = sess.run([network.acc, network.acc_op],
                               feed_dict={
                                   network.states: train_images[idx],
                                   network.labels: train_labels[idx][:, None],
                               })
        print(acc)
        print(acc_op)

        for i in range(10000):
            idx = np.random.choice(range(train_images.shape[0]), 64, replace=False)
            sess.run(network.apply_grad,
                     feed_dict={
                         network.states: train_images[idx],
                         network.labels: train_labels[idx][:, None],
                     })

        idx = np.random.choice(range(train_images.shape[0]), 64, replace=False)
        acc, acc_op = sess.run([network.acc, network.acc_op],
                               feed_dict={
                                   network.states: train_images[idx],
                                   network.labels: train_labels[idx][:, None],
                               })
        print(acc)
        print(acc_op)
        acc = sess.run(network.acc)
        print(acc)
