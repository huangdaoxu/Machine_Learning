import os
import collections

from config import *

PIC_WIDTH = 224
PIC_HEIGHT = 224
PIC_CHANNELS = 3

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
src_data = number + alphabet + ALPHABET


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "image",
                                           "label",
                                           "filenames"))):
    pass


class DataIterator(object):
    def __init__(self):
        ...

    def __read_tensor(self):
        self.filename = tf.placeholder(dtype=tf.string)
        self.suffix = tf.placeholder(dtype=tf.string)
        image_string = tf.read_file(self.filename)

        image_decoded = tf.case(
            pred_fn_pairs={tf.equal(self.suffix, 'jpg'): lambda: tf.image.decode_jpeg(image_string, channels=PIC_CHANNELS),
                           tf.equal(self.suffix, 'png'): lambda: tf.image.decode_png(image_string, channels=PIC_CHANNELS),
                           tf.equal(self.suffix, 'bmp'): lambda: tf.image.decode_bmp(image_string, channels=PIC_CHANNELS)},
            default=None,
            exclusive=True)
        image = tf.cast(image_decoded, tf.float32)
        resized_image = tf.image.resize_images(image, [PIC_HEIGHT, PIC_WIDTH])
        self.resized_image = tf.div(resized_image, 255.0)

    def __list_images(self, data_dir, unique_labels=src_data):
        file_path = []
        label_dict = {}
        for f in os.listdir(data_dir):
            file_path.append(os.path.join(data_dir, f))

        for i, label in enumerate(unique_labels):
            label_dict[label] = i
        return file_path, label_dict

    def image2tfrecord(self, data_dir, sess):
        self.__read_tensor()
        file_path, label_dict = self.__list_images(data_dir)
        writer = tf.python_io.TFRecordWriter(FLAGS.test_records_dir)
        for file in file_path:
            code = os.path.basename(file).split('.')[0]
            suffix = os.path.basename(file).split('.')[1]
            if len(code) != 4:
                continue
            image = sess.run(self.resized_image, feed_dict={self.filename: file,
                                                            self.suffix: suffix})
            label = [label_dict[c] for c in code]
            features = {}
            features['image'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
            features['label'] = tf.train.Feature(int64_list=tf.train.Int64List(value=label))
            tf_features = tf.train.Features(feature=features)
            tf_example = tf.train.Example(features=tf_features)
            tf_serialized = tf_example.SerializeToString()
            writer.write(tf_serialized)
        writer.close()

    def pares_tf(self, example_proto):
        dics = {}
        dics['label'] = tf.FixedLenFeature(shape=[4], dtype=tf.int64)
        dics['image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
        parsed_example = tf.parse_single_example(serialized=example_proto, features=dics)
        image = tf.decode_raw(parsed_example['image'], out_type=tf.float32)
        image = tf.reshape(image, shape=[224, 224, 3])
        label = parsed_example['label']
        label = tf.cast(label, tf.int64)
        return image, label

    def get_iterator(self, batch_size, buffer_size=None,
                     random_seed=None, num_threads=6):
        filenames = tf.placeholder(tf.string, shape=[None])
        if buffer_size is None:
            buffer_size = batch_size * 5

        dataset = tf.data.TFRecordDataset(filenames=filenames)
        dataset = dataset.map(self.pares_tf, num_parallel_calls=num_threads)
        dataset.prefetch(buffer_size)
        dataset = dataset.shuffle(buffer_size, random_seed)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_initializable_iterator()
        image, label = iterator.get_next()
        return BatchedInput(
            initializer=iterator.initializer,
            image=image,
            label=label,
            filenames=filenames)


if __name__ == "__main__":
    dataset = DataIterator()
    with tf.Session() as sess:
        dataset.image2tfrecord(FLAGS.test_pic_dir, sess)

    # dataset = DataIterator()
    # iterator = dataset.get_iterator(batch_size=32)
    # with tf.Session() as sess:
    #     sess.run(iterator.initializer, feed_dict={iterator.filenames: [FLAGS.train_records_dir]})
    #
    #     while True:
    #         try:
    #             image, label = sess.run([iterator.image, iterator.label])
    #             print(image.shape, label.shape)
    #         except tf.errors.OutOfRangeError:
    #             break

