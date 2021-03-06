import argparse
import logging

from ocrctc import OcrLstmCtc

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')


def train(arguments):
    # build a model
    cnn = OcrLstmCtc(True, arguments.model_path)
    cnn.train_model(arguments.batch_size,
                    arguments.keep_prob,
                    arguments.epochs,
                    arguments.data_size,
                    arguments.learning_rate,
                    arguments.train_data_path,
                    arguments.test_data_path,)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--data_size', default=500, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--model_path', default='./save_ctc/')
    parser.add_argument('--train_data_path', default='F:/ml_data/imgN/train')
    parser.add_argument('--test_data_path', default='F:/ml_data/imgN/val')
    parser.add_argument('--keep_prob', default=0.5)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    train(args)
