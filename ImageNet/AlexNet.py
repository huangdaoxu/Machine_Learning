import tensorflow as tf


def alexnet(inputs, keep_prob, num_classes):
    output = tf.layers.conv2d(inputs=inputs, filters=96, kernel_size=[11, 11],
                              strides=4, padding="valid", activation=tf.nn.relu, name="conv1")
    output = tf.layers.max_pooling2d(inputs=output, pool_size=[3, 3], strides=2, name='pool1')
    output = tf.nn.local_response_normalization(output)

    output = tf.layers.conv2d(inputs=output, filters=256, kernel_size=[5, 5],
                              strides=1, padding="same", activation=tf.nn.relu, name="conv2")
    output = tf.layers.max_pooling2d(inputs=output, pool_size=[3, 3], strides=2, name='pool2')
    output = tf.nn.local_response_normalization(output)

    output = tf.layers.conv2d(inputs=output, filters=384, kernel_size=[3, 3],
                              strides=1, padding="same", activation=tf.nn.relu, name="conv3")

    output = tf.layers.conv2d(inputs=output, filters=384, kernel_size=[3, 3],
                              strides=1, padding="same", activation=tf.nn.relu, name="conv4")

    output = tf.layers.conv2d(inputs=output, filters=256, kernel_size=[3, 3],
                              strides=1, padding="same", activation=tf.nn.relu, name="conv5")

    output = tf.layers.max_pooling2d(inputs=output, pool_size=[3, 3], strides=2, name='pool5')

    output = tf.reshape(output, [-1, 6 * 6 * 256])

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense6")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=4096, activation=tf.nn.relu, name="dense7")
    output = tf.nn.dropout(output, keep_prob=keep_prob)

    output = tf.layers.dense(inputs=output, units=num_classes, activation=tf.nn.relu, name="dense8")

    return output


if __name__ == "__main__":
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3], name="inputs")
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name="keep_prob")
    output = alexnet(inputs, keep_prob, 1000)
    print(output)

