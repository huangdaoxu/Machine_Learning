import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

"""
file resources config
"""
tf.app.flags.DEFINE_string("vocab_file", "./resource/vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("train_src_file", "./resource/source.txt", "train source file dir")
tf.app.flags.DEFINE_string("train_tgt_file", "./resource/target.txt", "train target file dir")
tf.app.flags.DEFINE_string("test_src_file", "./resource/source_test.txt", "test source file dir")
tf.app.flags.DEFINE_string("test_tgt_file", "./resource/target_test.txt", "test target file dir")
tf.app.flags.DEFINE_string("model_path", "./resource/save/", "model path")
tf.app.flags.DEFINE_string("word2vec_model_path", "./resource/vector.bin", "word to vector model path")
tf.app.flags.DEFINE_string("vocab_vector_file", "./resource/vocab_vector.npy", "vocab vector file")
tf.app.flags.DEFINE_string("model_pb_file", "./resource/abnormal_detection_model.pb", "converted model file")
tf.app.flags.DEFINE_string("tb_path", "./resource/tb/", "tensorboard file path")

"""
common config
"""
tf.app.flags.DEFINE_integer("num_steps", 100, "number of input string max length")
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_integer("embedding_size", 32, "vocab vector embedding size")
tf.app.flags.DEFINE_integer("batch_size", 2, "batch size")

"""
cpu core config
"""
tf.app.flags.DEFINE_integer("cpu_num", 8, "cpu core number")

"""
learning rate config
"""
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("decay_steps", 5, "decay steps")
tf.app.flags.DEFINE_float("decay_rate", 0.9, "decay rate")

"""
rnn layer config
"""
tf.app.flags.DEFINE_integer("num_layers", 1, "number of rnn layer")
tf.app.flags.DEFINE_integer("num_hidden", 16, "hidden layer output dimension")
tf.app.flags.DEFINE_float("input_keep_prob", 0.5, "input keep prob")
tf.app.flags.DEFINE_float("output_keep_prob", 0.5, "output keep prob")
tf.app.flags.DEFINE_float("state_keep_prob", 1.0, "state keep prob")
