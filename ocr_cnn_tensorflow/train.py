# -*- coding: utf-8 -*-

from captcha.image import ImageCaptcha
import cv2
import tensorflow as tf
# from matplotlib import pyplot as plt
import numpy as np
import random
from sklearn.cross_validation import train_test_split

PIC_WIDTH = 100
PIC_HEIGHT = 60
PIC_CHANNELS = 3

def train():
    image = ImageCaptcha(width=100,fonts=['/root/Symbol.ttf'])
    for i in range(0,5):
        for j in range(1000,10000):
            image.write('{0}'.format(j), './data/{0}_{1}.png'.format(j,i))

    index_dict = {0:np.array([1.0,0,0,0,0,0,0,0,0,0]),
                  1:np.array([0,1.0,0,0,0,0,0,0,0,0]),
                  2:np.array([0,0,1.0,0,0,0,0,0,0,0]),
                  3:np.array([0,0,0,1.0,0,0,0,0,0,0]),
                  4:np.array([0,0,0,0,1.0,0,0,0,0,0]),
                  5:np.array([0,0,0,0,0,1.0,0,0,0,0]),
                  6:np.array([0,0,0,0,0,0,1.0,0,0,0]),
                  7:np.array([0,0,0,0,0,0,0,1.0,0,0]),
                  8:np.array([0,0,0,0,0,0,0,0,1.0,0]),
                  9:np.array([0,0,0,0,0,0,0,0,0,1.0])}

    x_data = []
    y_data = []
    for i in range(0,4):
        for j in range(1000,10000):
            img = cv2.imread('./data/{0}_{1}.png'.format(j,i))
            x_data.append(img/255.0)
            y_data.append([index_dict[j/1000],index_dict[j%1000/100],index_dict[j%100/10],index_dict[j%10]])
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=123, test_size=0.1)
    del x_data,y_data
    print X_train.shape,X_test.shape

    #build a model

    x = tf.placeholder("float", shape=[None, PIC_HEIGHT,PIC_WIDTH,PIC_CHANNELS],name='x')
    y = tf.placeholder("float", shape=[None, 4, 10],name='y')
    keep_prob = tf.placeholder("float",name='keep_prob')

    W_conv1 = tf.Variable(tf.truncated_normal([3, 3, 3, 32], stddev=0.1),name='W_conv1')
    b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]),name='b_conv1')
    #第一层卷积激活池化
    h_conv1 = tf.nn.conv2d(x,W_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1
    h_relu1 = tf.nn.relu(h_conv1)
    h_pool1 = tf.nn.max_pool(h_relu1, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32], stddev=0.1),name='W_conv2')
    b_conv2 = tf.Variable(tf.constant(0.1, shape=[32]),name='b_conv2')
    #第二层卷积激活池化
    h_conv2 = tf.nn.conv2d(h_pool1,W_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2
    h_relu2 = tf.nn.relu(h_conv2)
    h_pool2 = tf.nn.max_pool(h_relu2, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 32, 64], stddev=0.1),name='W_conv3')
    b_conv3 = tf.Variable(tf.constant(0.1, shape=[64]),name='b_conv3')
    #第三层卷积激活池化
    h_conv3 = tf.nn.conv2d(h_pool2,W_conv3,strides=[1,1,1,1],padding='SAME') + b_conv3
    h_relu3 = tf.nn.relu(h_conv3)
    h_pool3 = tf.nn.max_pool(h_relu3, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

    print 'pool3',h_pool3.get_shape()

    W_fc1 = tf.Variable(tf.truncated_normal([8*13*64, 1024], stddev=0.1),name='W_fc1')
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024]),name='b_fc1')
    #reshape成全连接层，dropout避免过拟合
    h_fc1 = tf.matmul(tf.reshape(h_pool3, [-1, 8*13*64]), W_fc1) + b_fc1
    #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc21 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name='W_fc21')
    b_fc21 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_fc21')

    W_fc22 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name='W_fc22')
    b_fc22 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_fc22')

    W_fc23 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name='W_fc23')
    b_fc23 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_fc23')

    W_fc24 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1),name='W_fc24')
    b_fc24 = tf.Variable(tf.constant(0.1, shape=[10]),name='b_fc24')
    #分成四个全连接层，分别对应输出四位
    h_fc21 = tf.matmul(h_fc1, W_fc21) + b_fc21
    h_fc22 = tf.matmul(h_fc1, W_fc22) + b_fc22
    h_fc23 = tf.matmul(h_fc1, W_fc23) + b_fc23
    h_fc24 = tf.matmul(h_fc1, W_fc24) + b_fc24

    print 'h_fc21',h_fc21.get_shape()
    #整合结果
    y_conv = tf.stack([h_fc21,h_fc22,h_fc23,h_fc24], 1)

    print 'y_conv',y_conv.get_shape()
    print 'y',y.get_shape()
    #求损失函数
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))
    #求准确率
    correct_prediction = tf.equal(tf.argmax(y_conv, 2), tf.argmax(y, 2))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    epochs = 50
    batch_size = 200

    global_step = tf.Variable(0, trainable=False)
    #每10个迭代降一次学习率
    learning_rate = tf.train.exponential_decay(0.0001,
                                               global_step,
                                               (X_train.shape[0]/batch_size + 1)*10,
                                               0.96,
                                               staircase=True)

    train_step = tf.train.AdamOptimizer(0.0001).minimize(loss)

    train_index = range(X_train.shape[0])
    test_index = range(X_test.shape[0])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        for j in range(X_train.shape[0]/batch_size + 1):
            start = j*batch_size
            end = start + batch_size
            train_x, train_y = X_train[start:end], y_train[start:end]
            sess.run(train_step, feed_dict={x: train_x, y: train_y, keep_prob: 0.5})
        random.shuffle(train_index)
        random.shuffle(test_index)
        test_acc = sess.run(accuracy,
                            feed_dict={x: X_test[test_index[0:500]],
                                       y: y_test[test_index[0:500]],
                                       keep_prob: 1.0}
                            )
        train_acc = sess.run(accuracy,
                             feed_dict={x: X_train[train_index[0:500]],
                                        y: y_train[train_index[0:500]],
                                        keep_prob: 1.0}
                             )
        test_loss = sess.run(loss,
                             feed_dict={x: X_test[test_index[0:500]],
                                             y: y_test[test_index[0:500]],
                                             keep_prob: 1.0}
                             )
        train_loss = sess.run(loss,
                              feed_dict={x: X_train[train_index[0:500]],
                                         y: y_train[train_index[0:500]],
                                         keep_prob: 1.0}
                              )
        print 'epoch is %d,train_acc is %g,test_acc is %g!'%(i, train_acc,test_acc)
        print 'train_loss is %f,test_loss is %f'%(train_loss, test_loss)
    print 'end!!!'

    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, "./save/model.ckpt")





