import tensorflow as tf
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_pic_dir", "/home/hdx/train4data/train", "picture dir")
tf.app.flags.DEFINE_string("test_pic_dir", "/home/hdx/train4data/test", "picture dir")

tf.app.flags.DEFINE_string("train_records_dir", "./train.tfrecords", "picture dir")
tf.app.flags.DEFINE_string("test_records_dir", "./test.tfrecords", "picture dir")


