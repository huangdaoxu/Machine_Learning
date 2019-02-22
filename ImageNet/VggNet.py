import tensorflow as tf


def convolution(inputs, layer_num, filters,
                kernel_size, padding, activation):
    output = inputs
    for _ in range(layer_num):
        output = tf.layers.conv2d(inputs=output, filters=filters,
                                  kernel_size=kernel_size, padding=padding,
                                  activation=activation)
    return output


def vgg16(inputs, keep_prob, num_classes):
    with tf.name_scope('conv1'):
        output = convolution(inputs=inputs, layer_num=2, filters=64,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool1')

    with tf.name_scope('conv2'):
        output = convolution(inputs=output, layer_num=2, filters=128,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool2')

    with tf.name_scope('conv3'):
        output = convolution(inputs=output, layer_num=3, filters=256,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool3')

    with tf.name_scope('conv4'):
        output = convolution(inputs=output, layer_num=3, filters=512,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool4')

    with tf.name_scope('conv5'):
        output = convolution(inputs=output, layer_num=3, filters=512,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool5')

    output = tf.reshape(output, shape=[-1, 7 * 7 * 512])

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense6")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense7")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=num_classes, activation=tf.nn.relu, name="dense8")

    return output


def vgg19(inputs, keep_prob, num_classes):
    with tf.name_scope('conv1'):
        output = convolution(inputs=inputs, layer_num=2, filters=64,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool1')

    with tf.name_scope('conv2'):
        output = convolution(inputs=output, layer_num=2, filters=128,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool2')

    with tf.name_scope('conv3'):
        output = convolution(inputs=output, layer_num=4, filters=256,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool3')

    with tf.name_scope('conv4'):
        output = convolution(inputs=output, layer_num=4, filters=512,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool4')

    with tf.name_scope('conv5'):
        output = convolution(inputs=output, layer_num=4, filters=512,
                             kernel_size=[3, 3], padding='same', activation=tf.nn.relu)
        output = tf.layers.max_pooling2d(inputs=output, pool_size=[2, 2], strides=2, name='pool5')

    output = tf.reshape(output, shape=[-1, 7 * 7 * 512])

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense6")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense7")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=num_classes, activation=tf.nn.relu, name="dense8")

    return output


if __name__ == "__main__":
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="inputs")
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
    # output = vgg16(inputs, keep_prob, 1000)
    output = vgg19(inputs, keep_prob, 1000)
    print(output)

