from utils import *


class ConvertedModel(collections.namedtuple("ConvertedModel",
                                           ("source",
                                            "source_sequence_length",
                                            "y_pred",
                                            ))):
    pass


def save2pb(checkpoint_file, graph_file):
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(graph_file)
        saver.restore(sess, checkpoint_file)

        graph_def = tf.get_default_graph().as_graph_def()

        output_graph_def = tf.graph_util.convert_variables_to_constants(
            sess,
            graph_def,
            ["y_pred"]
        )

        with tf.gfile.GFile(FLAGS.model_pb_file, "wb") as f:
            f.write(output_graph_def.SerializeToString())


def load_pb(model_path):
    with tf.gfile.FastGFile(model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    y_pred, source, source_sequence_length = tf.import_graph_def(
        graph_def,
        return_elements=["y_pred:0", "source:0", "source_sequence_length:0"]
    )

    return ConvertedModel(
        source=source,
        source_sequence_length=source_sequence_length,
        y_pred=y_pred)


def pred_txt_data_by_pb():
    cm = load_pb(FLAGS.model_pb_file)
    iterator = get_pred_iterator(batch_size=FLAGS.batch_size)

    with tf.Session() as sess:
        tf.tables_initializer().run()
        sess.run(iterator.initializer,
                 feed_dict={iterator.input_source_file: FLAGS.test_src_file})

        pred_value = []

        while True:
            try:
                s, ssl = sess.run([iterator.source, iterator.source_sequence_length])
                pv = sess.run(cm.y_pred, feed_dict={cm.source: s,
                                                    cm.source_sequence_length: ssl})
                pred_value.extend(np.squeeze(pv).tolist())
            except tf.errors.OutOfRangeError:
                break

        true_value = []
        with open(FLAGS.test_tgt_file, encoding="utf-8") as f:
            for i in f.readlines():
                true_value.append(int(i))

        print((np.array(pred_value) == np.array(true_value)).mean())


def pred_txt_data_by_restore():
    iterator = get_pred_iterator(batch_size=FLAGS.batch_size)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(FLAGS.model_path + "point-20.meta")
        saver.restore(sess, FLAGS.model_path + "point-20")
        graph = tf.get_default_graph()
        source = graph.get_tensor_by_name("source:0")
        source_sequence_length = graph.get_tensor_by_name("source_sequence_length:0")
        y_pred = graph.get_operation_by_name("y_pred").outputs[0]

        tf.tables_initializer().run()
        sess.run(iterator.initializer,
                 feed_dict={iterator.input_source_file: FLAGS.test_src_file})

        pred_value = []

        while True:
            try:
                s, ssl = sess.run([iterator.source, iterator.source_sequence_length])
                pv = sess.run(y_pred, feed_dict={source: s,
                                                 source_sequence_length: ssl})
                pred_value.extend(np.squeeze(pv).tolist())
            except tf.errors.OutOfRangeError:
                break

        true_value = []
        with open(FLAGS.test_tgt_file, encoding="utf-8") as f:
            for i in f.readlines():
                true_value.append(int(i))

        print((np.array(pred_value) == np.array(true_value)).mean())


if __name__ == "__main__":
    save2pb("./resource/save/points-20", "./resource/save/points-20.meta")
    # pred_txt_data_by_pb()
    # pred_txt_data_by_restore()
