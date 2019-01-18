from utils import *


class Rnn_Net(object):
    def __init__(self, embedding_size, iterator):
        self.vocab_size = get_vocab_size(FLAGS.vocab_file)
        self.num_classes = get_num_classes(FLAGS.train_tgt_file)
        self.embedding_size = embedding_size
        # self.batch_size = batch_size
        self.iterator = iterator
        self.embeddings = load_word2vec_embedding(
            self.vocab_size,
            self.embedding_size
        )
        self._build_net()

    def _build_net(self):
        self.global_step = tf.Variable(0, trainable=False)
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)
        source = self.iterator.source
        target_input = self.iterator.target_input
        source_sequence_length = self.iterator.source_sequence_length

        rnn_inputs = tf.nn.embedding_lookup(params=self.embeddings, ids=source)

        stack = tf.nn.rnn_cell.MultiRNNCell(
            [self._get_lstm_cell(FLAGS.num_hidden) for _ in range(0, FLAGS.num_layers)],
            state_is_tuple=True
        )

        outputs, _ = tf.nn.dynamic_rnn(
            cell=stack,
            inputs=rnn_inputs,
            sequence_length=source_sequence_length,
            dtype=tf.float32
        )

        outputs = tf.reshape(outputs, [-1, FLAGS.num_steps * FLAGS.num_hidden])
        print(outputs.get_shape())

        # outputs = tf.layers.dense(outputs, units=512,
        #                           activation=tf.nn.relu, kernel_regularizer=regularizer)
        #
        # outputs = tf.layers.dense(outputs, units=64,
        #                           activation=tf.nn.relu, kernel_regularizer=regularizer)

        logits = tf.layers.dense(outputs, units=self.num_classes,
                                 activation=None, kernel_regularizer=regularizer)
        print(logits.get_shape())

        self.y_pred = tf.argmax(logits, 1)

        tf.losses.sparse_softmax_cross_entropy(labels=target_input, logits=logits)
        self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        self.train_op = tf.train.AdamOptimizer(
            learning_rate=FLAGS.learning_rate).minimize(self.loss)

    @staticmethod
    def _get_lstm_cell(num_units):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units,
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
            state_is_tuple=True
        )
        return cell


def train(net, iterator, sess):
    sess.run(tf.global_variables_initializer())
    tf.tables_initializer().run()

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        saver.restore(sess, path)
    current_epoch = sess.run(net.global_step)
    while True:
        if current_epoch > FLAGS.epoch: break
        sess.run(iterator.initializer)
        while True:
            try:
                losses, _ = sess.run([net.loss, net.train_op])
                print('train loss :', losses)
            except tf.errors.OutOfRangeError:
                break
        if current_epoch % 10 == 0:
            saver.save(sess, FLAGS.model_path + 'points', global_step=current_epoch)

        current_epoch += 1
        sess.run(tf.assign(net.global_step, current_epoch))
        print('current_epoch', current_epoch)
