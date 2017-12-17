# coding: utf-8
"""
Created on Tue Oct 24 15:26:09 2017

@author: hdx
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_data = pd.read_json('data/processed/train.json')
train_data.inc_angle = train_data.inc_angle.replace('na',0)
train_data.inc_angle = train_data.inc_angle.astype(np.float32)

def bn(xs, on_train):
   fc_mean, fc_var = tf.nn.moments(
       xs,
       axes=[0],
   )
   scale = tf.Variable(tf.ones([1]))
   shift = tf.Variable(tf.zeros([1]))
   epsilon = 0.001
   # apply moving average for mean and var when train on batch
   ema = tf.train.ExponentialMovingAverage(decay=0.5)
   def mean_var_with_update():
       ema_apply_op = ema.apply([fc_mean, fc_var])
       with tf.control_dependencies([ema_apply_op]):
           return tf.identity(fc_mean), tf.identity(fc_var)
   #mean, var = mean_var_with_update()
   mean, var = tf.cond(on_train,    # on_train 的值是 True/False
                   mean_var_with_update,   # 如果是 True, 更新 mean/var
                   lambda: (               # 如果是 False, 返回之前 fc_mean/fc_var 的Moving Average
                       ema.average(fc_mean), 
                       ema.average(fc_var)
                       )    
                   )
   return tf.nn.batch_normalization(xs, mean, var, shift, scale, epsilon)

#构建神经网络模型
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 75,75,3],name='x')
y = tf.placeholder("float", shape=[None, 1],name='y')
keep_prob = tf.placeholder("float",name='keep_prob')
on_train = tf.placeholder("bool",name='on_train')

W_conv1 = weight_variable([3, 3, 3, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#h_pool1_frop = tf.nn.dropout(h_pool1, keep_prob)

W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#h_pool2_frop = tf.nn.dropout(h_pool2, keep_prob)

W_conv3 = weight_variable([3, 3, 64, 64])
b_conv3 = bias_variable([64])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)

h_pool3_drop = tf.nn.dropout(h_pool3, keep_prob)

print 'pool3',h_pool3.get_shape()

W_fc1 = weight_variable([10*10*64, 64])
b_fc1 = bias_variable([64])

h_pool3_flat = tf.reshape(h_pool3_drop, [-1, 10*10*64])
#h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
#h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([64, 64])
b_fc2 = bias_variable([64])

h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

W_fc3 = weight_variable([64, 1])
b_fc3 = bias_variable([1])

h_fc3 = tf.matmul(h_fc2_drop, W_fc3) + b_fc3

y_conv = tf.sigmoid(h_fc3)
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_conv)))
tf.summary.scalar('loss',rmse)
#y_conv = tf.matmul(h_fc2_drop, W_fc3) + b_fc3
#rmse = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_conv))

#求准确率
#correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

from sklearn import preprocessing

def Rotate(data,k):
    rotated_data = []
    T_rotated_data = []
    for i in data[['band_1','band_2','inc_angle']].values:
        tmp_band_1 = np.rot90(np.array(i[0]).reshape(75,75),k)
        tmp_band_2 = np.rot90(np.array(i[1]).reshape(75,75),k)
        tmp_band_half = np.rot90((tmp_band_1 + tmp_band_2)/2,k)
        tmp_band_1 = preprocessing.scale(tmp_band_1)
        tmp_band_2 = preprocessing.scale(tmp_band_2)
        tmp_band_half = preprocessing.scale(tmp_band_half)
        rotated_data.append(np.transpose(np.array([tmp_band_1,tmp_band_2,tmp_band_half]),(2,1,0)))
        T_rotated_data.append(np.transpose(np.array([tmp_band_1.T,tmp_band_2.T]),(2,1,0)))
        #rotated_data.append(np.array([tmp_band_1,tmp_band_2,tmp_band_half]).reshape(75,75,3))
        #T_rotated_data.append(np.array([tmp_band_1.T,tmp_band_2.T,tmp_band_half.T]).reshape(75,75,3))
    return np.array(rotated_data),np.array(T_rotated_data)

from sklearn.cross_validation import train_test_split

X_data,T_X_data = Rotate(train_data, 0)
X90_data,T_X90_data = Rotate(train_data, 1)
X180_data,T_X180_data = Rotate(train_data, 2)
X270_data,T_X270_data = Rotate(train_data, 3)
y_data = train_data[['is_iceberg']].values
#y_data = []
#for i in train_data[['is_iceberg']].values:
#    y_data.append(([1,0] if i[0]==0 else [0,1]))

rotated_total_X_data = np.concatenate((X_data, X90_data, X180_data, X270_data),axis=0)
rotated_total_y_data = np.concatenate((y_data, y_data, y_data, y_data),axis=0)

X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, random_state=123, test_size=0.2)

print X_train.shape,X_test.shape

del X_data,T_X_data,X90_data,T_X90_data,X180_data,T_X180_data,X270_data,T_X270_data,rotated_total_X_data,rotated_total_y_data

import random

train_index = range(X_train.shape[0])
test_index = range(X_test.shape[0])

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
#learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                           7, 0.96, staircase=True)

train_step = tf.train.AdamOptimizer(starter_learning_rate).minimize(rmse, global_step=global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./test/",sess.graph)

train_loss = []
test_loss = []

epochs = 100
batch_size = 32

for i in range(epochs):
    for j in range(X_train.shape[0]/batch_size + 1):
        start = j*batch_size
        end = start + batch_size
        train_x, train_y = X_train[start:end], y_train[start:end]
        sess.run(train_step, feed_dict={x: train_x, y: train_y, keep_prob: 0.5, on_train: True})
    random.shuffle(train_index)
    random.shuffle(test_index)
    #test_validation_loss = sess.run(merged, feed_dict={x: X_test, y: y_test, keep_prob: 1.0, on_train: False})   
    #writer.add_summary(loss, i) #result是summary类型的，需要放入writer中，i步数（x轴）    
    #train_validation_loss = sess.run(merged, feed_dict={x: X_train, y: y_train, keep_prob: 1.0, on_train: False})
    #writer.add_summary(train_validation_loss,i) #result是summary类型的，需要放入writer中，i步数（x轴）  
    test_validation_loss = sess.run(rmse, feed_dict={x: X_test, y: y_test, keep_prob: 1.0, on_train: False})
    train_validation_loss = sess.run(rmse, feed_dict={x: X_train, y: y_train, keep_prob: 1.0, on_train: False})
    #test_validation_loss = sess.run(rmse, feed_dict={x: X_test[test_index[0:500]], y: y_test[test_index[0:500]], keep_prob: 1.0})
    #train_validation_loss = sess.run(rmse, feed_dict={x: X_train[train_index[0:500]], y: y_train[train_index[0:500]], keep_prob: 1.0})
    print 'epoch is {0},test_validation_loss is {1},train_validation_loss is {2}!'.format(i, test_validation_loss, train_validation_loss)
    #if abs(test_validation_loss-train_validation_loss)>0.1:
    #    break
print 'end!!!'

test_data = pd.read_json('data/processed/test.json')
rotated_test_data , _ = Rotate(test_data,0)

prediction_list = []
for i in rotated_test_data:
    prediction_list.append(sess.run(y_conv, feed_dict={x: np.array([i]), keep_prob:1.0}))

test_data['is_iceberg'] = 0
for i in range(len(prediction_list)):
    test_data.loc[i, 'is_iceberg'] = prediction_list[i][0][0]
    
test_data[['id','is_iceberg']].to_csv('predicted_data.csv', index=None)



