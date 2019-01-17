import os

from rnn_model import *


if __name__ == '__main__':
    if not os.path.exists(FLAGS.word2vec_model_path):
        raise OSError('word2vec model file does not exist')

    iterator = get_iterator(buffer_size=FLAGS.batch_size, batch_size=FLAGS.batch_size, random_seed=666)

    net = Rnn_Net(
        embedding_size=FLAGS.embedding_size,
        iterator=iterator
    )

    with tf.Session() as sess:
        train(net, iterator, sess)

