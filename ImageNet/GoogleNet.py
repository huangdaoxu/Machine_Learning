import tensorflow as tf


def inception_v1(inputs, num_classes):
    with tf.name_scope(name="Conv2d_1a_7x7"):
        net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[7, 7],
                               strides=2, padding="same", activation=tf.nn.relu)

    with tf.name_scope(name="MaxPool_2a_3x3"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                      strides=2, padding='same')

    with tf.name_scope(name="Conv2d_2b_1x1"):
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1],
                               strides=1, padding="same", activation=tf.nn.relu)

    with tf.name_scope(name="Conv2d_2c_3x3"):
        net = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[3, 3],
                               strides=1, padding="same", activation=tf.nn.relu)

    with tf.name_scope(name="MaxPool_3a_3x3"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                      strides=2, padding='same')

    with tf.name_scope(name="Mixed_3b"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=96, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=128, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=32, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=32, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_3c"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=192, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=96, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="MaxPool_4a_3x3"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                      strides=2, padding='same')

    with tf.name_scope(name="Mixed_4b"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=96, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=208, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=16, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=48, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_4c"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=112, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=224, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=24, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_4d"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=256, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=24, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_4e"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=112, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=144, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=288, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=64, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=64, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_4f"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=256, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=320, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="MaxPool_5a_2x2"):
        net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2],
                                      strides=2, padding='same')

    with tf.name_scope(name="Mixed_5b"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=256, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=160, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=320, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding='same')
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    with tf.name_scope(name="Mixed_5c"):
        with tf.name_scope(name="branch_0"):
            branch_0 = tf.layers.conv2d(inputs=net, filters=384, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_1"):
            branch_1 = tf.layers.conv2d(inputs=net, filters=192, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_1 = tf.layers.conv2d(inputs=branch_1, filters=384, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_2"):
            branch_2 = tf.layers.conv2d(inputs=net, filters=48, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)
            branch_2 = tf.layers.conv2d(inputs=branch_2, filters=128, kernel_size=[3, 3],
                                        strides=1, padding="same", activation=tf.nn.relu)

        with tf.name_scope(name="branch_3"):
            branch_3 = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                               strides=1, padding="same")
            branch_3 = tf.layers.conv2d(inputs=branch_3, filters=128, kernel_size=[1, 1],
                                        strides=1, padding="same", activation=tf.nn.relu)

        net = tf.concat([branch_0, branch_1, branch_2, branch_3], 3)

    net = tf.layers.average_pooling2d(inputs=net, pool_size=[7, 7],
                                      strides=1, padding="valid")
    net = tf.squeeze(net, [1, 2])
    net = tf.layers.dense(inputs=net, units=num_classes)
    return net


if __name__ == "__main__":
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="inputs")
    net = inception_v1(inputs, 1000)
    print(net)
