import sys
sys.path.append("../")

from utils import *
from ImageNet.GoogleNet import inception_v1


class Ocr4LenCnnModel(object):
    def __init__(self, is_training, model_path, ):
        self.is_training = is_training
        self.model_path = model_path
        self.__build_model()

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
        self.x = tf.placeholder("float", shape=[None, PIC_HEIGHT, PIC_WIDTH, PIC_CHANNELS], name='x')
        self.y = tf.placeholder(tf.int64, shape=[None, 4], name='y')
        self.keep_prob = tf.placeholder("float", name='keep_prob')
        self.global_step = tf.Variable(0, trainable=False)
        self.training = tf.placeholder(tf.bool, name='training')

        net = inception_v1(self.x, 4*num_classes)
        logits = tf.reshape(net, [-1, num_classes, 4])

        tf.losses.sparse_softmax_cross_entropy(labels=y, logits=logits)
        self.loss = tf.losses.get_total_loss(add_regularization_losses=True, name='total_loss')

        self.y_pred = tf.argmax(logits, 1)
        self.correct_prediction = tf.equal(self.y_pred, self.y)

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


if __name__ == "__main__":
    cnn = Ocr4LenCnnModel(True, "./save/")

