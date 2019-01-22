import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("vocab_file", "./vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("train_src_file", "./source.txt", "train source file dir")
tf.app.flags.DEFINE_string("train_tgt_file", "./target.txt", "train target file dir")
tf.app.flags.DEFINE_string("model_path", "./save/", "model path")
tf.app.flags.DEFINE_string("word2vec_model_path", "./vector.bin", "word to vector model path")
tf.app.flags.DEFINE_string("vocab_vector_file", "./vocab_vector.npy", "vocab vector file")
tf.app.flags.DEFINE_integer("num_hidden", 16, "hidden layer output dimension")
tf.app.flags.DEFINE_integer("num_layers", 1, "number of rnn layer")
tf.app.flags.DEFINE_integer("num_steps", 100, "number of input string max length")
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_integer("embedding_size", 32, "vocab vector embedding size")
tf.app.flags.DEFINE_integer("batch_size", 2, "batch size")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
