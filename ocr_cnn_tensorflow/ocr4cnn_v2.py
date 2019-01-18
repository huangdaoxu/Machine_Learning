from ocr4_util import *


class Ocr4LenCnnModel(object):
    def __init__(self, is_training, model_path, ):
        self.is_training = is_training
        self.model_path = model_path
        (self.x, self.y, self.keep_prob, self.loss, self.y_pred,
         self.training, self.global_step, self.correct_prediction) = self.__build_model()

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
    def batch_norm(x, is_training, name):
        bn = tf.layers.batch_normalization(
            inputs=x,
            axis=-1,
            momentum=0.9,
            epsilon=0.001,
            center=True,
            scale=True,
            training=is_training,
            name=name
        )
        return bn

    def __build_model(self):
        num_classes = len(src_data)
        x = tf.placeholder("float", shape=[None, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNELS], name='x')
        y = tf.placeholder(tf.int64, shape=[None, 4], name='y')
        keep_prob = tf.placeholder("float", name='keep_prob')
        global_step = tf.Variable(0, trainable=False)
        training = tf.placeholder(tf.bool, name='training')

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001)

        with tf.name_scope('conv1'):
            out = tf.layers.conv2d(inputs=x, filters=32, kernel_size=[3, 3], name='conv11',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out, filters=32, kernel_size=[3, 3], name='conv12',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = self.batch_norm(out, training, name='conv1_bn')
            out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool1')

        with tf.name_scope('conv2'):
            out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], name='conv21',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], name='conv22',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            # out = tf.layers.conv2d(inputs=out, filters=64, kernel_size=[3, 3], name='conv23',
            #                        padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = self.batch_norm(out, training, name='conv2_bn')
            out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=[1, 2], name='pool2', padding='same')

        with tf.name_scope('conv3'):
            out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], name='conv31',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], name='conv32',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            # out = tf.layers.conv2d(inputs=out, filters=128, kernel_size=[3, 3], name='conv33',
            #                        padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = self.batch_norm(out, training, name='conv3_bn')
            out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool3')

        with tf.name_scope('conv4'):
            out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], name='conv41',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], name='conv42',
                                   padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            # out = tf.layers.conv2d(inputs=out, filters=256, kernel_size=[3, 3], name='conv43',
            #                        padding='same', kernel_regularizer=regularizer, activation=tf.nn.relu)
            out = self.batch_norm(out, training, name='conv4_bn')
            out = tf.layers.max_pooling2d(inputs=out, pool_size=[2, 2], strides=2, name='pool4')

        out = tf.layers.conv2d(inputs=out, filters=2048, kernel_size=[7, 7], name='conv5', padding='valid',
                               activation=tf.nn.relu, kernel_regularizer=regularizer)
        out = self.batch_norm(out, training, name='conv5_bn')
        out = tf.nn.dropout(out, keep_prob)
        print(out.get_shape())
        out = tf.reshape(out, [-1, 2048], name='conv5/reshape')

        out11 = tf.layers.dense(out, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out12 = tf.layers.dense(out, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out13 = tf.layers.dense(out, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out14 = tf.layers.dense(out, units=1024, activation=tf.nn.relu, kernel_regularizer=regularizer)

        out11 = self.batch_norm(out11, training, name='out11_bn')
        out12 = self.batch_norm(out12, training, name='out12_bn')
        out13 = self.batch_norm(out13, training, name='out13_bn')
        out14 = self.batch_norm(out14, training, name='out14_bn')

        out21 = tf.layers.dense(out11, units=512, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out22 = tf.layers.dense(out12, units=512, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out23 = tf.layers.dense(out13, units=512, activation=tf.nn.relu, kernel_regularizer=regularizer)
        out24 = tf.layers.dense(out14, units=512, activation=tf.nn.relu, kernel_regularizer=regularizer)

        out21 = self.batch_norm(out21, training, name='out21_bn')
        out22 = self.batch_norm(out22, training, name='out22_bn')
        out23 = self.batch_norm(out23, training, name='out23_bn')
        out24 = self.batch_norm(out24, training, name='out24_bn')

        out31 = tf.layers.dense(out21, units=num_classes, activation=None, kernel_regularizer=regularizer)
        out32 = tf.layers.dense(out22, units=num_classes, activation=None, kernel_regularizer=regularizer)
        out33 = tf.layers.dense(out23, units=num_classes, activation=None, kernel_regularizer=regularizer)
        out34 = tf.layers.dense(out24, units=num_classes, activation=None, kernel_regularizer=regularizer)

        logits = tf.stack([out31, out32, out33, out34], 1)

        tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        y_pred = tf.argmax(logits, 2)
        correct_prediction = tf.equal(y_pred, y)

        return x, y, keep_prob, loss, y_pred, training, global_step, correct_prediction

    def check_accuracy(self, data_set, sess):
        """
        Check the accuracy of the model on either train or val (depending on dataset_init_op).
        """
        # Initialize the correct dataset
        data_set.data_init_op(sess)
        num_correct, num_samples = 0, 0
        while True:
            try:
                x, y = data_set.get_next()
                if x is None: break
                correct_pred = sess.run(self.correct_prediction,
                                        feed_dict={self.x: x,
                                                   self.y: y,
                                                   self.keep_prob: 1.0,
                                                   self.training: False})
                num_correct += correct_pred.sum()
                num_samples += correct_pred.shape[0] * 4
            except tf.errors.OutOfRangeError:
                break

        # Return the fraction of datapoints that were correctly classified
        acc = float(num_correct) / num_samples
        return acc

    def train_model(self, train_data_size, test_data_size, batch_size, decay_steps, keep_prob,
                    epoch, learning_rate, train_data_dir, test_data_dir):
        if not self.is_training:
            raise ValueError("this is compute mode, please reconstruct model with is_training")

        train_dataset = DataIterator(data_size=train_data_size, batch_size=batch_size, data_dir=train_data_dir)
        test_dataset = DataIterator(data_size=test_data_size, batch_size=batch_size, data_dir=test_data_dir)

        lr = tf.train.exponential_decay(
            learning_rate=learning_rate,
            global_step=self.global_step,
            decay_steps=decay_steps,
            decay_rate=0.96,
            staircase=False,
            name='learning_rate'
        )

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(lr).minimize(self.loss)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.5
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())

            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(self.model_path)
            if ckpt is not None:
                path = ckpt.model_checkpoint_path
                saver.restore(sess, path)

            current_epoch = sess.run(self.global_step)
            current_epoch += 1

            while True:
                if current_epoch > epoch: break
                train_dataset.data_init_op(sess)
                while True:
                    train_x, train_y = train_dataset.get_next()
                    if train_x is None: break
                    _ = sess.run(train_step, feed_dict={self.x: train_x,
                                                        self.y: train_y,
                                                        self.keep_prob: keep_prob,
                                                        self.training: True})

                if current_epoch % 10 == 0:
                    train_acc = self.check_accuracy(train_dataset, sess)
                    val_acc = self.check_accuracy(test_dataset, sess)
                    logging.info('epoch: {}, Train accuracy: {}, Val accuracy: {}'.format(current_epoch, train_acc, val_acc))
                    sess.run(tf.assign(self.global_step, current_epoch))
                    saver.save(sess, self.model_path + 'points', global_step=current_epoch)

                current_epoch += 1

    def compute(self, picture=None, suffix=None):
        if self.is_training:
            raise ValueError("this is training mode, please reconstruct model without is_training")
        with tf.Session() as sess:
            if suffix == 'png':
                image_decoded = tf.image.decode_png(picture, channels=PIC_CHANNELS)
            elif suffix == 'jpg':
                image_decoded = tf.image.decode_jpeg(picture, channels=PIC_CHANNELS)
            elif suffix == 'bmp':
                image_decoded = tf.image.decode_bmp(picture, channels=PIC_CHANNELS)
            else:
                raise ValueError("file format is not supported")
            image_decoded = tf.cast(image_decoded, tf.float32)
            resized_image = tf.image.resize_images(image_decoded, [PIC_HEIGHT, PIC_WIDTH])
            resized_image = tf.div(resized_image, 255.0)
            resized_image = sess.run(resized_image)

        result = self.sess.run(self.y_pred, feed_dict={self.x: [resized_image],
                                                       self.keep_prob: 1.0,
                                                       self.training: False})

        result_str = ''
        for i in result[0]:
            result_str += src_data[i]
        return result_str
