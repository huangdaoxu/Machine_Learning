import tensorflow as tf
import logging

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("train_pic_dir", "/home/hdx/train4data/train", "picture dir")
tf.app.flags.DEFINE_string("test_pic_dir", "/home/hdx/train4data/test", "picture dir")
tf.app.flags.DEFINE_string("train_records_dir", "./train.tfrecords", "picture dir")
tf.app.flags.DEFINE_string("test_records_dir", "./test.tfrecords", "picture dir")
tf.app.flags.DEFINE_string("model_path", "./resource/save/", "model path")

tf.app.flags.DEFINE_integer("epochs", 200, "number of training epoch")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_integer("decay_steps", 5, "decay steps of learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.96, "decay rate of learning rate")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.5, "keep prob")

