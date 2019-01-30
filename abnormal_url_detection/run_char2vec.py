import os
import pandas as pd

from char2vec import Char2Vec
from utils import *
from config import FLAGS


if __name__ == "__main__":
    primary_df = pd.read_csv('./primary.csv')
    source_list = primary_df['request_url'].values.tolist()
    print(primary_df['request_url'].str.len())
    target_list = primary_df['status'].values.tolist()
    source_list = parse_http_request(source_list)
    source_list = write_src_tgt(source_list, target_list,
                                FLAGS.train_src_file, FLAGS.train_tgt_file)

    model = Char2Vec(size=FLAGS.embedding_size, window=5, min_count=1, workers=2, iter=5)
    model_path = FLAGS.word2vec_model_path
    if os.path.exists(model_path):
        model.load(model_path)
    else:
        model.train(sentences=source_list)
        model.save(model_path)
    model.write_vocab(FLAGS.vocab_file, FLAGS.vocab_vector_file)
