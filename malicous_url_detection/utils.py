import numpy as np
import tensorflow as tf

import collections

from urllib import parse
from config import FLAGS


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target_input",
                                           "source_sequence_length"))):
    pass


def create_vocab_tables(file):
    vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=file, num_oov_buckets=1)
    return vocab_table


def get_vocab_size(file):
    with open(file, encoding='utf-8') as f:
        count = len(f.readlines())
    return count


def get_num_classes(file):
    with open(file, encoding='utf-8') as f:
        count = len(set([i.strip('\n') for i in f.readlines()]))
    return count


def parse_http_request(http_req_list):
    parsed_req_list = []
    for http_req in http_req_list:
        decoded = parse.unquote(http_req)
        parsed = parse.urlparse(decoded)
        if parsed.path == '':
            continue
        if parsed.query == '':
            parsed_req_list.append(parsed.path)
        else:
            parsed_req_list.append(parsed.path + '?' + parsed.query)
    return parsed_req_list


def write_src_tgt(src_list, tgt_list, src_file, tgt_file):
    char_list = [[j for j in i] for i in src_list]
    with open(src_file, mode='w+', encoding='utf-8') as f:
        for i in char_list:
            f.write(' '.join(i) + '\n')
        src_lines = len(f.readlines())

    with open(tgt_file, mode='w+', encoding='utf-8') as f:
        for i in tgt_list:
            f.write(str(i) + '\n')
        tgt_lines = len(f.readlines())

    if src_lines != tgt_lines:
        raise ValueError('line number was different with write source and target in files')

    return char_list


def load_word2vec_embedding(vocab_size, embeddings_size):
    '''
        加载外接的词向量。
        :return:
    '''
    print('loading word embedding, it will take few minutes...')
    embeddings = np.random.uniform(-1, 1, (vocab_size + 2, embeddings_size))
    # 保证每次随机出来的数一样。
    rng = np.random.RandomState(666)
    unknown = np.asarray(rng.normal(size=embeddings_size))
    padding = np.asarray(rng.normal(size=embeddings_size))

    vocab_vector = np.load(FLAGS.vocab_vector_file)

    for index, value in enumerate(vocab_vector):
        embeddings[index] = value
    # 顺序不能错，这个和unkown_id和padding id需要一一对应。
    embeddings[-2] = unknown
    embeddings[-1] = padding

    return tf.get_variable("embeddings", dtype=tf.float32,
                           shape=[vocab_size + 2, embeddings_size],
                           initializer=tf.constant_initializer(embeddings), trainable=False)


def get_iterator(batch_size, buffer_size=None, random_seed=None,
                 num_threads=4, src_max_len=FLAGS.num_steps, num_buckets=5):
    vocab_table = create_vocab_tables(FLAGS.vocab_file)
    vocab_size = get_vocab_size(FLAGS.vocab_file)

    if buffer_size is None:
        buffer_size = batch_size * 5

    src_dataset = tf.data.TextLineDataset(FLAGS.train_src_file)
    tgt_dataset = tf.data.TextLineDataset(FLAGS.train_tgt_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    # split data
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_to_number(tf.string_split([tgt]).values, out_type=tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # get max len data
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (src[:src_max_len], tgt),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # vocab table look up
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt, tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # calculate text line true length
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt_in: (
            src, tgt_in, tf.size(src)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([src_max_len]),  # src
                           tf.TensorShape([1]),  # tgt_input
                           tf.TensorShape([])),  # src_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(vocab_size + 1,  # src
                            0,  # tgt_input
                            0))  # src_len -- unused

    def key_func(unused_1, unused_2, src_len):
        if src_max_len:
            bucket_width = (src_max_len + num_buckets - 1) // num_buckets
        else:
            bucket_width = 10

        bucket_id = src_len // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    # batched_dataset = src_tgt_dataset.apply(tf.contrib.data.group_by_window(
    #     key_func=key_func, reduce_func=reduce_func, window_size=batch_size
    # ))
    batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_ids, src_seq_len) = (batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target_input=tgt_ids,
        source_sequence_length=src_seq_len)


if __name__ == '__main__':
    #################### Just for testing #########################
    iterator = get_iterator(batch_size=6, random_seed=666)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        for i in range(1):
            sess.run(iterator.initializer)
            while True:
                try:
                    source, target_input, source_sequence_length = \
                        sess.run([iterator.source, iterator.target_input,
                                  iterator.source_sequence_length])
                    print(source)
                    print(target_input)
                    print(source_sequence_length)
                except tf.errors.OutOfRangeError:
                    break

