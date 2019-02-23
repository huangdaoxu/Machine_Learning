import argparse

from ocr4cnn_v3 import Ocr4LenCnnModel


def train(arguments):
    # build a model
    cnn = Ocr4LenCnnModel(True, arguments.model_path)
    cnn.train_model(arguments.batch_size,
                    arguments.decay_steps,
                    arguments.keep_prob,
                    arguments.epochs,
                    arguments.learning_rate)


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--decay_steps', default=5, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--model_path', default='./save4/')
    parser.add_argument('--keep_prob', default=0.4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()
    train(args)
