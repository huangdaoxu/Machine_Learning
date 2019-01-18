import argparse
import logging

from ocr4cnn_v2 import Ocr4LenCnnModel

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


def train(arguments):
    # build a model
    cnn = Ocr4LenCnnModel(True, arguments.model_path)
    cnn.train_model(arguments.train_data_size,
                    arguments.test_data_size,
                    arguments.batch_size,
                    arguments.decay_steps,
                    arguments.keep_prob,
                    arguments.epochs,
                    arguments.learning_rate,
                    arguments.train_data_path,
                    arguments.test_data_path)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--train_data_size', default=500, type=int)
    parser.add_argument('--test_data_size', default=500, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--decay_steps', default=5, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--model_path', default='./save4/')
    parser.add_argument('--train_data_path', default='F:/ml_data/train4data/train')
    parser.add_argument('--test_data_path', default='F:/ml_data/train4data/test')
    parser.add_argument('--keep_prob', default=0.4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    train(args)
