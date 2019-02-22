import tensorflow as tf


def block_v1(inputs, filters, strides, training):
    shortcut = tf.layers.conv2d(inputs=inputs, filters=4*filters, kernel_size=[1, 1],
                                strides=strides, padding="same", activation=None)
    shortcut = batch_norm(shortcut, training)

    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[1, 1],
                              strides=1, padding="same", activation=None)

    inputs = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=[3, 3],
                              strides=strides, padding="same", activation=None)

    inputs = tf.layers.conv2d(inputs=inputs, filters=4*filters, kernel_size=[1, 1],
                              strides=1, padding="same", activation=None)
    inputs = batch_norm(inputs, training)
    inputs += shortcut
    inputs = tf.nn.relu(inputs)

    return inputs


def batch_norm(inputs, training):
  return tf.layers.batch_normalization(
      inputs=inputs, momentum=0.997, epsilon=1e-5, center=True,
      scale=True, training=training, fused=True)


def resnet_50(inputs, training, num_classes):
    net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[7, 7],
                           strides=2, padding="same", activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                  strides=2, padding='same')
    for _ in range(3):
        net = block_v1(net, 64, 1, training)
    net = block_v1(net, 128, 2, training)
    for _ in range(3):
        net = block_v1(net, 128, 1, training)
    net = block_v1(net, 256, 2, training)
    for _ in range(5):
        net = block_v1(net, 256, 1, training)
    net = block_v1(net, 512, 2, training)
    for _ in range(2):
        net = block_v1(net, 512, 1, training)

    net = tf.layers.average_pooling2d(inputs=net, pool_size=[7, 7],
                                      strides=1, padding="valid")
    net = tf.squeeze(net, [1, 2])
    net = tf.layers.dense(inputs=net, units=num_classes)
    return net


def resnet_101(inputs, training, num_classes):
    net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[7, 7],
                           strides=2, padding="same", activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                  strides=2, padding='same')
    for _ in range(3):
        net = block_v1(net, 64, 1, training)
    net = block_v1(net, 128, 2, training)
    for _ in range(3):
        net = block_v1(net, 128, 1, training)
    net = block_v1(net, 256, 2, training)
    for _ in range(22):
        net = block_v1(net, 256, 1, training)
    net = block_v1(net, 512, 2, training)
    for _ in range(2):
        net = block_v1(net, 512, 1, training)

    net = tf.layers.average_pooling2d(inputs=net, pool_size=[7, 7],
                                      strides=1, padding="valid")
    net = tf.squeeze(net, [1, 2])
    net = tf.layers.dense(inputs=net, units=num_classes)
    return net


def resnet_152(inputs, training, num_classes):
    net = tf.layers.conv2d(inputs=inputs, filters=64, kernel_size=[7, 7],
                           strides=2, padding="same", activation=tf.nn.relu)

    net = tf.layers.max_pooling2d(inputs=net, pool_size=[3, 3],
                                  strides=2, padding='same')
    for _ in range(3):
        net = block_v1(net, 64, 1, training)
    net = block_v1(net, 128, 2, training)
    for _ in range(3):
        net = block_v1(net, 128, 1, training)
    net = block_v1(net, 256, 2, training)
    for _ in range(22):
        net = block_v1(net, 256, 1, training)
    net = block_v1(net, 512, 2, training)
    for _ in range(2):
        net = block_v1(net, 512, 1, training)

    net = tf.layers.average_pooling2d(inputs=net, pool_size=[7, 7],
                                      strides=1, padding="valid")
    net = tf.squeeze(net, [1, 2])
    net = tf.layers.dense(inputs=net, units=num_classes)
    return net


if __name__ == "__main__":
    inputs = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name="inputs")
    training = tf.placeholder(dtype=tf.bool, shape=[], name="training")
    net = resnet_152(inputs, training, 1000)
    print(net)
