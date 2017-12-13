"""
Created on 2017-12-13 23:05

@author: huangdaoxu
"""

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import matplotlib.pyplot as plt


class autoencoder(object):
    def __init__(self):
        self._X = tf.placeholder(tf.float32, shape=[None,784], name='X')
        # encoder params
        self._en_w1 = tf.Variable(tf.truncated_normal([784, 64], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_w1')
        self._en_b1 = tf.Variable(tf.truncated_normal([64], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_b1')
        self._en_w2 = tf.Variable(tf.truncated_normal([64, 32], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_w2')
        self._en_b2 = tf.Variable(tf.truncated_normal([32], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_b2')
        self._en_w3 = tf.Variable(tf.truncated_normal([32, 16], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_w3')
        self._en_b3 = tf.Variable(tf.truncated_normal([16], mean=0.0, stddev=1.0, dtype=tf.float32), name='en_b3')
        # decoder params
        self._de_w1 = tf.Variable(tf.truncated_normal([16, 32], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_w1')
        self._de_b1 = tf.Variable(tf.truncated_normal([32], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_b1')
        self._de_w2 = tf.Variable(tf.truncated_normal([32, 64], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_w2')
        self._de_b2 = tf.Variable(tf.truncated_normal([64], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_b2')
        self._de_w3 = tf.Variable(tf.truncated_normal([64, 784], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_w3')
        self._de_b3 = tf.Variable(tf.truncated_normal([784], mean=0.0, stddev=1.0, dtype=tf.float32), name='de_b3')


    def encode(self):
        # activation is important ,we must choice a right function.
        en_result1 = tf.nn.sigmoid(tf.matmul(self._X, self._en_w1) + self._en_b1)
        en_result2 = tf.nn.sigmoid(tf.matmul(en_result1, self._en_w2) + self._en_b2)
        en_result3 = tf.nn.sigmoid(tf.matmul(en_result2, self._en_w3) + self._en_b3)
        return en_result3

    def decode(self, encode):
        # activation is important ,we must choice a right function.
        de_result1 = tf.nn.sigmoid(tf.matmul(encode, self._de_w1) + self._de_b1)
        de_result2 = tf.nn.sigmoid(tf.matmul(de_result1, self._de_w2) + self._de_b2)
        de_result3 = tf.nn.sigmoid(tf.matmul(de_result2, self._de_w3) + self._de_b3)
        return de_result3

    def train(self, epoch=5, batch_size=256):
        y_pred = self.decode(self.encode())
        loss = tf.reduce_mean(tf.square(self._X - y_pred))
        optmizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        sess = tf.Session()

        sess.run(tf.global_variables_initializer())

        total_batch = int(mnist.train.num_examples / batch_size)
        for i in xrange(epoch):
            for j in xrange(total_batch):
                batch_xs, _ = mnist.train.next_batch(batch_size)
                sess.run(optmizer, feed_dict={self._X: batch_xs})
            print 'epoch is {0},loss is {1}'.format(i, sess.run(loss, feed_dict={self._X: batch_xs}))

        data_pred = sess.run(y_pred, feed_dict={self._X: mnist.test.images[0:10]})
        f, a = plt.subplots(2, 10, figsize=(10, 2))
        for j in xrange(data_pred.shape[0]):
            a[0][j].imshow(mnist.test.images[j].reshape(28,28))
            a[1][j].imshow(data_pred[j].reshape(28,28))
        plt.show()




if __name__ == "__main__":
    ae = autoencoder()
    ae.train()