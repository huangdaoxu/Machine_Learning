import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

"""
file resources config
"""
tf.app.flags.DEFINE_string("vocab_file", "./vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("train_src_file", "./source.txt", "train source file dir")
tf.app.flags.DEFINE_string("train_tgt_file", "./target.txt", "train target file dir")
tf.app.flags.DEFINE_string("test_src_file", "./source_test.txt", "test source file dir")
tf.app.flags.DEFINE_string("test_tgt_file", "./target_test.txt", "test target file dir")
tf.app.flags.DEFINE_string("model_path", "./save/", "model path")
tf.app.flags.DEFINE_string("word2vec_model_path", "./vector.bin", "word to vector model path")
tf.app.flags.DEFINE_string("vocab_vector_file", "./vocab_vector.npy", "vocab vector file")

"""
common config
"""
tf.app.flags.DEFINE_integer("num_hidden", 16, "hidden layer output dimension")
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
tf.app.flags.DEFINE_float("decay_rate", 0.96, "decay rate")

"""
rnn layer config
"""
tf.app.flags.DEFINE_integer("num_layers", 2, "number of rnn layer")
tf.app.flags.DEFINE_float("input_keep_prob", 0.7, "input keep prob")
tf.app.flags.DEFINE_float("output_keep_prob", 0.7, "output keep prob")
tf.app.flags.DEFINE_float("state_keep_prob", 0.7, "state keep prob")
