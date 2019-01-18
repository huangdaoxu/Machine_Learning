import random
import logging
import os

import numpy as np
import tensorflow as tf

from PIL import Image
from captcha.image import ImageCaptcha

PIC_WIDTH = 120
PIC_HEIGHT = 60
PIC_CHANNELS = 3
FONT_SIZES = [i for i in range(30, 40)]

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

src_data = number + alphabet + ALPHABET


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
        self.__list_images(unique_labels=src_data)

    def __build_model(self):
        self.filename = tf.placeholder(dtype=tf.string)
        self.suffix = tf.placeholder(dtype=tf.string)
        image_string = tf.read_file(self.filename)

        def readerror():
            raise ValueError('read batch data error, suffix mismatched')

        image_decoded = tf.case(
            pred_fn_pairs={tf.equal(self.suffix, 'jpg'): lambda: tf.image.decode_jpeg(image_string, channels=PIC_CHANNELS),
                           tf.equal(self.suffix, 'png'): lambda: tf.image.decode_png(image_string, channels=PIC_CHANNELS),
                           tf.equal(self.suffix, 'bmp'): lambda: tf.image.decode_bmp(image_string, channels=PIC_CHANNELS)},
            default=None,
            exclusive=True)
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
            code = os.path.basename(file).split('.')[0]
            suffix = os.path.basename(file).split('.')[1]
            if len(code) != 4:
                continue
            self.imgs.append(sess.run(self.resized_image, feed_dict={self.filename: file, self.suffix: suffix}))
            code = [self.label_dict[c] for c in code]
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


def random_captcha_text(char_set=src_data, captcha_size=4):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


def str_2_one_hot(txt, char_set=src_data):
    one_hot_index = []
    one_hot_len = len(char_set)
    for i in txt:
        one_hot_index.append(src_data.index(i))
    one_hot_encode = np.eye(one_hot_len)[one_hot_index]
    return one_hot_encode


def gen_captcha_text_and_image(width=PIC_WIDTH, height=PIC_HEIGHT, font_sizes=FONT_SIZES):
    image = ImageCaptcha(width=width, height=height, font_sizes=font_sizes)
    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text)
    captcha = image.generate(captcha_text)
    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    one_hot_encode = str_2_one_hot(captcha_text)
    return captcha_image, one_hot_encode


def get_batch_data(batch_size=256):
    x = []
    y = []
    for i in range(0, batch_size):
        a, b = gen_captcha_text_and_image()
        x.append(a)
        y.append(b)
    return np.asarray(x) / 255.0, np.asarray(y)
