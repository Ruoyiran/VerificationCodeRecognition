'''
@version: 1.0
@author: royran
@contact: iranpeng@gmail.com
@file: dataset.py
@time: 2018/2/2 20:25
'''

import tensorflow as tf
from tf_utils import TfRecordReaderHelper

class DataSet(object):
    def __init__(self, tfrecord_file_path):
        self._file_path = tfrecord_file_path

    def next_batch(self, batch_size, shuffle=True):
        reader = TfRecordReaderHelper()
        img_obj = reader.load_image_object(self._file_path)
        if img_obj is None:
            raise IOError("failed to load file '{}'".format(self._file_path))
        image_raw = tf.decode_raw(img_obj.image_raw, tf.uint8)
        digits_raw = tf.decode_raw(img_obj.digits, tf.int32)

        height = tf.cast(img_obj.height, tf.int32)
        width = tf.cast(img_obj.width, tf.int32)
        image = tf.reshape(image_raw, [60, 160, 3])
        digits = tf.cast(digits_raw, tf.int32)
        digits = tf.reshape(digits, [4])

        image = tf.cast(image, tf.float32)
        image = tf.divide(image, 255.0)

        num_threads = 1

        if shuffle:
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * batch_size
            image_batch, label_batch = tf.train.shuffle_batch([image, digits],
                                                              batch_size=batch_size,
                                                              capacity=capacity,
                                                              min_after_dequeue=min_after_dequeue,
                                                              num_threads=num_threads)
        else:
            capacity=10000 + 3 * batch_size
            image_batch, label_batch = tf.train.batch([image, digits],
                                                      batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      capacity=capacity)
        return image_batch, label_batch
