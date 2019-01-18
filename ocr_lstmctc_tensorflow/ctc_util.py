import random
import os
import cv2
import logging

import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha

PIC_WIDTH = 180
PIC_HEIGHT = 60
PIC_CHANNELS = 3

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

CHARS = number + alphabet + ALPHABET
VEC_NUM = len(CHARS)+1+1  # blank + ctc blank

# 图片显示字符个数
MIN_LABEL_LENGTH = 2
MAX_LABEL_LENGTH = 8

SPACE_INDEX = 0
SPACE_TOKEN = ''
FIRST_INDEX = 1


class DataIterator(object):
    def __init__(self, data_size, batch_size, data_dir):
        self.data_size = data_size
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.file_path = []
        self.label_dict = {}
        self.imgs = []
        self.y = []
        self.index = 0
        self.__build_model()
        self.__list_images(unique_labels=CHARS)

    def __build_model(self):
        self.filename = tf.placeholder(dtype=tf.string)
        image_string = tf.read_file(self.filename)
        image_decoded = tf.image.decode_jpeg(image_string, channels=PIC_CHANNELS)
        image = tf.cast(image_decoded, tf.float32)
        resized_image = tf.image.resize_images(image, [PIC_HEIGHT, PIC_WIDTH])
        self.resized_image = tf.div(resized_image, 255.0)

    def __list_images(self, unique_labels):
        """
        Get all the images and labels in directory
        """
        for f in os.listdir(self.data_dir):
            self.file_path.append(os.path.join(self.data_dir, f))

        for i, label in enumerate(unique_labels):
            self.label_dict[label] = i

        logging.info('total data size {}'.format(len(self.file_path)))
        logging.info('batch {}'.format(self.data_size // self.batch_size))

    def data_init_op(self, sess):
        self.imgs = []
        self.y = []
        self.index = 0
        slice_path = random.sample(self.file_path, self.data_size)
        random.shuffle(slice_path)

        for file in slice_path:
            self.imgs.append(sess.run(self.resized_image, feed_dict={self.filename: file}))
            code = os.path.basename(file).split('.')[0]
            code = [SPACE_INDEX if c == SPACE_TOKEN else (self.label_dict[c] + FIRST_INDEX) for c in code]
            self.y.append(code)

        self.imgs = np.asarray(self.imgs)
        self.y = np.asarray(self.y)
        logging.info('slice image data shape {},slice y data shape {}'.format(self.imgs.shape, self.y.shape))

    def get_next(self):
        if len(self.imgs) <= 0:
            raise ValueError('please do data_init_op func before this func')
        start = self.index * self.batch_size
        end = (self.index + 1) * self.batch_size
        x = None
        y = None
        if self.index < (self.data_size // self.batch_size):
            x, y = self.imgs[start:end], self.y[start:end]
            self.index += 1
        return x, y


def create_dataset(total_size=5):
    """
    创建数据集
    """
    image = ImageCaptcha(width=PIC_WIDTH, height=PIC_HEIGHT, fonts=None, font_sizes=[i for i in range(30, 40)])

    def gen_rand():
        """
        随机获取字符串
        """
        buf = ""
        max_len = random.randint(MIN_LABEL_LENGTH, MAX_LABEL_LENGTH)
        for i in range(max_len):
            buf += random.choice(CHARS)
        return buf

    for i in range(total_size):
        chars = gen_rand()
        data = image.generate_image(chars)
        data.save("F:/ml_data/imgN/train/" + chars + ".jpg")


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    设置tensorflow稀疏变量
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int64)

    return indices, values, shape


def read_data(img_dir):
    image = []
    labels = []

    label_to_int = {}
    for i, label in enumerate(CHARS):
        label_to_int[label] = i

    for filename in os.listdir(img_dir):
        img = cv2.imread(os.path.join(img_dir, filename), cv2.IMREAD_COLOR).astype(np.float32) / 255.
        img = np.reshape(img, [PIC_HEIGHT, PIC_WIDTH, PIC_CHANNELS])
        image.append(img)

        code = filename.split('.')[0]
        code = [SPACE_INDEX if c == SPACE_TOKEN else (label_to_int[c] + FIRST_INDEX) for c in code]
        labels.append(code)
    return np.array(image), np.array(labels)


# create_dataset(50000)
