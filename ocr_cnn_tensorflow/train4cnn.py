from ocr4cnn_v4 import Ocr4LenCnnModel


if __name__ == '__main__':
    cnn = Ocr4LenCnnModel(True)
    cnn.train_model()
