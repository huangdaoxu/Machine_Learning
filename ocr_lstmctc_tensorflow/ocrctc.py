from common.tool.ctc_util import *


class OcrLstmCtc(object):
    def __init__(self, is_training, model_path, ):
        self.is_training = is_training
        self.model_path = model_path
        (self.input_x, self.labels, self.loss,
         self.global_step, self.max_length,
         self.dense_decoded, self.cost, self.keep_prob, self.seq_len) = self.__build_model()

        if not is_training:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                saver.restore(self.sess, path)
            else:
                raise ValueError("model path is not find")

    @staticmethod
    def list_images(unique_labels, directory):
        """
        Get all the images and labels in directory
        """
        files_and_labels = []
        for f in os.listdir(directory):
            files_and_labels.append((os.path.join(directory, f), os.path.splitext(f)[0]))

        filenames, labels = zip(*files_and_labels)
        filenames = list(filenames)
        labels = list(labels)

        label_to_int = {}
        for i, label in enumerate(unique_labels):
            label_to_int[label] = i

        label_list = []

        for i in labels:
            label_list.append([SPACE_INDEX if c == SPACE_TOKEN else (label_to_int[c] + FIRST_INDEX) for c in i])

        return filenames, label_list

    @staticmethod
    def parse_function(filename, label, new_height=PIC_HEIGHT, new_width=PIC_WIDTH):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=3)
        image = tf.cast(image_decoded, tf.float32)

        resized_image = tf.image.resize_images(image, [new_height, new_width])
        return tf.div(resized_image, 255.0), label

    def batch_norm(self, x, name):
        bn = tf.layers.batch_normalization(
            inputs=x,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=self.is_training,
            name=name
        )
        return bn

    def __build_model(self):
        num_hidden = 256
        num_layers = 3
        num_classes = VEC_NUM

        input_x = tf.placeholder("float", shape=[None, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNELS], name='input_x')
        labels = tf.sparse_placeholder(tf.int32, name='labels')
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')
        global_step = tf.Variable(0, trainable=False)
        keep_prob = tf.placeholder("float", name='keep_prob')

        conv1 = tf.layers.conv2d(inputs=input_x, filters=64, kernel_size=[3, 3], padding="same", activation=None)
        conv11 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=None)
        conv1_bn = self.batch_norm(conv11, name='conv1_bn')
        conv1_act = tf.nn.relu(conv1_bn)
        pool1 = tf.layers.max_pooling2d(inputs=conv1_act, pool_size=[2, 2], strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv21 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv2_bn = self.batch_norm(conv21, name='conv2_bn')
        conv2_act = tf.nn.relu(conv2_bn)
        pool2 = tf.layers.max_pooling2d(inputs=conv2_act, pool_size=[2, 2], strides=2)

        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv31 = tf.layers.conv2d(inputs=conv3, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv3_bn = self.batch_norm(conv31, name='conv3_bn')
        conv3_act = tf.nn.relu(conv3_bn)
        pool3 = tf.layers.max_pooling2d(inputs=conv3_act, pool_size=[2, 2], strides=2)

        conv4 = tf.layers.conv2d(inputs=pool3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv41 = tf.layers.conv2d(inputs=conv4, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv4_bn = self.batch_norm(conv41, name='conv4_bn')
        conv4_act = tf.nn.relu(conv4_bn)
        pool4 = tf.layers.max_pooling2d(inputs=conv4_act, pool_size=[2, 2], strides=[2, 1])
        pool4_drop = tf.nn.dropout(pool4, keep_prob)

        dense_shape = pool4_drop.get_shape().as_list()
        pool4_tp = tf.transpose(pool4_drop, [0, 2, 1, 3])

        cnn_out = tf.reshape(pool4_tp, shape=[-1, dense_shape[2], dense_shape[1] * dense_shape[3]])
        print(cnn_out.get_shape())
        max_length = dense_shape[2]

        def get_cell(num_hidden):
            cell = tf.nn.rnn_cell.LSTMCell(num_hidden,
                                           initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1),
                                           state_is_tuple=True)
            return cell

        stack = tf.contrib.rnn.MultiRNNCell([get_cell(num_hidden) for _ in range(0, num_layers)], state_is_tuple=True)
        outputs, _ = tf.nn.dynamic_rnn(cell=stack,
                                       inputs=cnn_out,
                                       sequence_length=seq_len,
                                       dtype=tf.float32)

        shape = tf.shape(cnn_out)
        batch_s, max_timesteps = shape[0], shape[1]

        outputs = tf.reshape(outputs, [-1, num_hidden])

        W = tf.Variable(tf.truncated_normal([num_hidden, num_classes], stddev=0.1), name='W')
        b = tf.Variable(tf.constant(0.1, shape=[num_classes], name='b'))

        logits = tf.matmul(outputs, W) + b
        logits = tf.reshape(logits, [batch_s, -1, num_classes])
        logits = tf.transpose(logits, (1, 0, 2))

        loss = tf.nn.ctc_loss(labels=labels, inputs=logits, sequence_length=seq_len)
        cost = tf.reduce_mean(loss)

        decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len, merge_repeated=False)

        dense_decoded = tf.sparse_tensor_to_dense(decoded[0], default_value=-1)

        # acc = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32), labels))

        return input_x, labels, loss, global_step, max_length, dense_decoded, cost, keep_prob, seq_len

    @staticmethod
    def accuracy_calculation(original_seq, decoded_seq, ignore_value=-1, isPrint=False):
        if len(original_seq) != len(decoded_seq):
            print('original lengths is different from the decoded_seq, please check again')
            return 0
        count = 0
        for i, origin_label in enumerate(original_seq):
            decoded_label = [j for j in decoded_seq[i] if j != ignore_value]
            # if isPrint and i < 100:
            #     print('seq{0:4d}: origin: {1} decoded:{2}'.format(i, origin_label, decoded_label))
            if origin_label == decoded_label:
                count += 1
        return count * 1.0 / len(original_seq)

    def check_loss_and_accuracy(self, dataset, sess):
        loss_list = []
        acc_list = []
        dataset.data_init_op(sess)
        while True:
            x, y = dataset.get_next()
            if x is None: break
            seq = [self.max_length for _ in range(x.shape[0])]
            loss, dense_decoded = sess.run(
                [self.cost, self.dense_decoded],
                feed_dict={self.input_x: x, self.labels: sparse_tuple_from(y), self.seq_len: seq, self.keep_prob: 1.0})
            acc = self.accuracy_calculation(y, dense_decoded, ignore_value=-1, isPrint=True)
            loss_list.append(loss)
            acc_list.append(acc)
        return np.asarray(loss_list).mean(), np.asarray(acc_list).mean()

    def train_model(self, batch_size, keep_prob,
                    epoch, data_size, learning_rate,
                    train_data_dir, test_data_dir):
        if not self.is_training:
            raise ValueError("this is compute mode, please reconstruct model with is_training")

        train_dataset = DataIterator(data_size=data_size, batch_size=batch_size, data_dir=train_data_dir)
        test_dataset = DataIterator(data_size=data_size, batch_size=batch_size, data_dir=test_data_dir)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                saver.restore(sess, path)

            current_epoch = sess.run(self.global_step)

            while True:
                if current_epoch >= epoch: break
                train_dataset.data_init_op(sess)
                while True:
                    train_x, train_y = train_dataset.get_next()
                    if train_x is None: break
                    train_y = sparse_tuple_from(train_y)
                    seq = [self.max_length for _ in range(train_x.shape[0])]
                    sess.run(train_step, feed_dict={self.input_x: train_x,
                                                    self.labels: train_y,
                                                    self.seq_len: seq,
                                                    self.keep_prob: keep_prob})

                if (current_epoch + 1) % 10 == 0:
                    train_loss, train_acc = self.check_loss_and_accuracy(train_dataset, sess)
                    test_loss, test_acc = self.check_loss_and_accuracy(test_dataset, sess)

                    print('epoch is %d,train_loss is %f,test_loss is %f' % (current_epoch + 1, train_loss, test_loss))
                    print('epoch is %d,train_acc is %f,test_acc is %f' % (current_epoch + 1, train_acc, test_acc))
                    saver.save(sess, self.model_path + 'points', global_step=current_epoch + 1)

                sess.run(tf.assign(self.global_step, current_epoch + 1))
                current_epoch += 1

    def compute(self, picture=None, suffix=None):
        if self.is_training:
            raise ValueError("this is training mode, please reconstruct model without is_training")
        with tf.Session() as new_sess:
            if suffix == 'png':
                image_decoded = tf.image.decode_png(picture, channels=PIC_CHANNELS)
            elif suffix == 'jpg':
                image_decoded = tf.image.decode_jpeg(picture, channels=PIC_CHANNELS)
            else:
                raise ValueError("file format is not supported")
            image_decoded = tf.cast(image_decoded, tf.float32)
            resized_image = tf.image.resize_images(image_decoded, [PIC_HEIGHT, PIC_WIDTH])
            resized_image = tf.div(resized_image, 255.)
            resized_image = new_sess.run(resized_image)

        seq = [self.max_length]
        result = self.sess.run(self.dense_decoded, feed_dict={self.input_x: [resized_image], self.seq_len: seq})
        result_str = ''
        for i in result[0]:
            result_str += CHARS[i - FIRST_INDEX]
        return result_str
