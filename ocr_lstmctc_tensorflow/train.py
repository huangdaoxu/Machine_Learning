# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import random
from captcha.image import ImageCaptcha
from sklearn.model_selection import train_test_split


# 图片尺寸
PIC_WIDTH = 100
PIC_HEIGHT = 60
PIC_CHANNELS = 3

# 图片显示字符个数
MIN_LABEL_LENGTH = 4
MAX_LABEL_LENGTH = 6

# 额外设置
SPACE_INDEX = 0
SPACE_TOKEN = ''
FIRST_INDEX = 1

# 字符范围，暂时只做0-9的训练
DIGITS = "0123456789"
# LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHARS = list(DIGITS)
VEC_NUM = len(CHARS)+1+1  # blank + ctc blank

# 字体文件需要自己找到合适的
image = ImageCaptcha(width=PIC_WIDTH,fonts=['/root/Symbol.ttf'])


def gen_rand():
    """
    随机获取字符串
    """
    buf = ""
    max_len = random.randint(MIN_LABEL_LENGTH, MAX_LABEL_LENGTH)
    for i in range(max_len):
        buf += random.choice(DIGITS)
    return buf


def get_vec(index, vec_num=VEC_NUM):
    """
    获取字符所对应的稀疏矩阵编码
    """
    a = np.zeros(vec_num)
    a[index] = 1
    return a


def create_dataset(total_size=5):
    """
    创建数据集
    """
    imgs = []
    labels = []
    for i in xrange(total_size):
        chars = gen_rand()
        # image.write(chars, './data/{0}_{1}.png'.format(chars,i))
        data=image.generate_image(chars)
        imgs.append(np.array(data.convert('L')))
        labels.append([SPACE_INDEX if c == SPACE_TOKEN else (CHARS.index(c) + FIRST_INDEX) for c in list(chars)])
    return np.array(imgs),np.array(labels)


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    设置tensorflow稀疏变量
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape

# 创建数据集并进行验证集与训练集的切分
imgs,labels = create_dataset(10000)
X_train, X_test, y_train, y_test = train_test_split(imgs/255., labels, random_state=123, test_size=0.1)
del imgs, labels
print X_train.shape, X_test.shape, y_train.shape, y_test.shape

# build model
num_hidden = 64
num_layers = 3
num_classes = VEC_NUM  # 十位数字+blank

inputs = tf.placeholder(tf.float32, [None, None, PIC_WIDTH],name='inputs')
labels = tf.sparse_placeholder(tf.int32,name='labels')
# labels = tf.sparse_placeholder(tf.float32, [None, num_classes],name='labels')
seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)
stack = tf.contrib.rnn.MultiRNNCell([cell for i in range(0, num_layers)], state_is_tuple=True)
outputs, _ = tf.nn.dynamic_rnn(cell, inputs, seq_len, dtype=tf.float32)

shape = tf.shape(inputs)
batch_s, max_timesteps = shape[0], shape[1]
outputs = tf.reshape(outputs, [-1, num_hidden])

W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name='W')
b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))

logits = tf.matmul(outputs, W) + b
logits = tf.reshape(logits, [batch_s, -1, num_classes])
logits = tf.transpose(logits, (1, 0, 2))
print logits.get_shape()
loss = tf.reduce_mean(tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_len))

decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len,merge_repeated=False)

epochs = 300
batch_size = 64

train_index = range(X_train.shape[0])
test_index = range(X_test.shape[0])

optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(epochs):
    for j in range(X_train.shape[0]/batch_size):
        start = j*batch_size
        end = start + batch_size
        train_x, train_y = X_train[start:end], y_train[start:end]
        seq = [PIC_HEIGHT for _ in range(train_x.shape[0])]
        labels_target = sparse_tuple_from(train_y)
        sess.run(optimizer, feed_dict={inputs: train_x, labels: labels_target, seq_len: seq})
    random.shuffle(train_index)
    random.shuffle(test_index)
    loss_bach = 64
    seq = [PIC_HEIGHT for _ in range(loss_bach)]
    test_loss = sess.run(loss,feed_dict={inputs: X_test[test_index[0:loss_bach]],
                                         labels: sparse_tuple_from(y_test[test_index[0:loss_bach]]),
                                         seq_len: seq})
    train_loss = sess.run(loss, feed_dict={inputs: X_train[train_index[0:loss_bach]],
                                           labels: sparse_tuple_from(y_train[train_index[0:loss_bach]]),
                                           seq_len: seq})
    print 'epoch is %d,train_loss is %f,test_loss is %f'%(i, train_loss, test_loss)

