from utils import *
from sklearn.metrics import confusion_matrix


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
            [self._get_rnn_dropout(self._get_lstm_cell(FLAGS.num_hidden)) for _ in range(0, FLAGS.num_layers)],
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

        logits = tf.layers.dense(outputs, units=self.num_classes,
                                 activation=None, kernel_regularizer=regularizer)
        print(logits.get_shape())

        self.y_pred = tf.cast(tf.argmax(logits, 1), tf.int32, name="y_pred")

        self.accurancy = tf.reduce_mean(tf.cast(tf.equal(self.y_pred, target_input), tf.float32))

        tf.losses.sparse_softmax_cross_entropy(labels=target_input, logits=logits)
        self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        self.learning_rate = tf.train.exponential_decay(
            learning_rate=FLAGS.learning_rate,
            global_step=FLAGS.epoch,
            decay_steps=FLAGS.decay_steps,
            decay_rate=FLAGS.decay_rate,
        )
        self.train_op = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate).minimize(self.loss)

    @staticmethod
    def _get_lstm_cell(num_units):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units,
            initializer=tf.orthogonal_initializer(),
            state_is_tuple=True,
            activation=tf.nn.tanh,
            forget_bias=1.0,
        )
        return cell

    @staticmethod
    def _get_rnn_dropout(cell):
        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=FLAGS.input_keep_prob,
            output_keep_prob=FLAGS.output_keep_prob,
            state_keep_prob=FLAGS.state_keep_prob,
        )
        return cell


def train(net, sess):
    sess.run(tf.global_variables_initializer())
    tf.tables_initializer().run()
    writer = tf.summary.FileWriter(FLAGS.tb_path, sess.graph)

    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(FLAGS.model_path)
    if ckpt is not None:
        path = ckpt.model_checkpoint_path
        saver.restore(sess, path)
    current_epoch = sess.run(net.global_step)
    while True:
        if current_epoch > FLAGS.epoch: break
        sess.run(net.iterator.initializer,
                 feed_dict={net.iterator.input_source_file: FLAGS.train_src_file,
                            net.iterator.input_target_file: FLAGS.train_tgt_file})
        while True:
            try:
                sess.run(net.train_op)
            except tf.errors.OutOfRangeError:
                break
        check_accuracy(sess, net, current_epoch, 'train')
        check_accuracy(sess, net, current_epoch, 'test')
        if current_epoch % 10 == 0:
            saver.save(sess, FLAGS.model_path + 'points', global_step=current_epoch)

        current_epoch += 1
        sess.run(tf.assign(net.global_step, current_epoch))

    writer.close()


def check_accuracy(sess, net, current_epoch, op_type):
    src_file = FLAGS.train_src_file if op_type == 'train' else FLAGS.test_src_file
    tgt_file = FLAGS.train_tgt_file if op_type == 'train' else FLAGS.test_tgt_file
    sess.run(net.iterator.initializer,
             feed_dict={net.iterator.input_source_file: src_file,
                        net.iterator.input_target_file: tgt_file})
    loss = []
    accurancy = []
    pred_value = []
    true_value = []
    while True:
        try:
            los, acc, pv, tv = sess.run([net.loss, net.accurancy,
                                         net.y_pred, net.iterator.target_input])
            loss.append(los)
            accurancy.append(acc)
            pred_value.extend(np.squeeze(pv).tolist())
            true_value.extend(np.squeeze(tv).tolist())
        except tf.errors.OutOfRangeError:
            print('data type :{}, current_epoch :{}, loss :{:.3}, accurancy :{:.3}, '
                  'false positive rate :{:.3}, false negative rate :{:.3}'.format(
                   op_type,
                   current_epoch,
                   sum(loss) / len(loss),
                   sum(accurancy) / len(accurancy),
                   false_positive_rate(true_value, pred_value),
                   false_negative_rate(true_value, pred_value),
            ))
            break


# 误报率
def false_positive_rate(true_value, pred_value):
    cm = confusion_matrix(true_value, pred_value)
    return cm[0][1]/(cm[0][1] + cm[1][1])


# 漏报率
def false_negative_rate(true_value, pred_value):
    cm = confusion_matrix(true_value, pred_value)
    return cm[1][0] / (cm[1][0] + cm[0][0])


