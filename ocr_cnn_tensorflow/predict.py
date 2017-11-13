import tensorflow as tf
import cv2

PIC_WIDTH = 100
PIC_HEIGHT = 60
PIC_CHANNELS = 3


class CnnModel(object):
    def __init__(self):
        self.x = 0
        self.y = 0
        self.keep_prob = 0.0
        self.sess = 0
        self.result = 0
        self.loss = 0.0
        self.accuracy = 0.0
        self.ConstructModel()

    def ConstructModel(self):
        # build a model

        self.x = tf.placeholder("float", shape=[None, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNELS], name='x')
        self.y = tf.placeholder("float", shape=[None, 4, 10], name='y')
        self.keep_prob = tf.placeholder("float", name='keep_prob')

        W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1), name='W_conv1')
        b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv1')

        h_conv1 = tf.nn.conv2d(self.x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
        h_relu1 = tf.nn.relu(h_conv1)
        h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1), name='W_conv2')
        b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]), name='b_conv2')

        h_conv2 = tf.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2
        h_relu2 = tf.nn.relu(h_conv2)
        h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1), name='W_conv3')
        b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]), name='b_conv3')

        h_conv3 = tf.nn.conv2d(h_pool2, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3
        h_relu3 = tf.nn.relu(h_conv3)
        h_pool3 = tf.nn.max_pool(h_relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        print 'pool3', h_pool3.get_shape()

        W_fc1 = tf.Variable(tf.truncated_normal([8 * 13 * 64, 1024], stddev=0.1), name='W_fc1')
        b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]), name='b_fc1')

        h_fc1 = tf.matmul(tf.reshape(h_pool3, [-1, 8 * 13 * 64]), W_fc1) + b_fc1
        # h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        W_fc21 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc21')
        b_fc21 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc21')

        W_fc22 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc22')
        b_fc22 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc22')

        W_fc23 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc23')
        b_fc23 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc23')

        W_fc24 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1), name='W_fc24')
        b_fc24 = tf.Variable(tf.constant(0.1, shape=[10]), name='b_fc24')

        h_fc21 = tf.matmul(h_fc1, W_fc21) + b_fc21
        h_fc22 = tf.matmul(h_fc1, W_fc22) + b_fc22
        h_fc23 = tf.matmul(h_fc1, W_fc23) + b_fc23
        h_fc24 = tf.matmul(h_fc1, W_fc24) + b_fc24

        print 'h_fc21', h_fc21.get_shape()

        y_conv = tf.stack([h_fc21, h_fc22, h_fc23, h_fc24], 1)

        print 'y_conv', y_conv.get_shape()
        print 'y', self.y.get_shape()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))

        correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(self.y, 2))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        saver = tf.train.Saver()
        self.sess = tf.Session()
        saver.restore(self.sess, "./save/model.ckpt")
        self.result = tf.argmax(y_conv, 2)

    def Calculate(self,photo_path):
        img = cv2.imread(photo_path)
        img = cv2.resize(img, (100, 60), interpolation=cv2.INTER_CUBIC)
        return self.sess.run(self.result, feed_dict={self.x: [img], self.keep_prob: 1.0})



