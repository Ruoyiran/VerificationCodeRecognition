'''
@version: 1.0
@author: royran
@contact: iranpeng@gmail.com
@file: load_data.py
@time: 2018/2/2 20:25
'''
import numpy as np
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

        height = tf.cast(img_obj.height, tf.int32)
        width = tf.cast(img_obj.width, tf.int32)
        image = tf.reshape(image_raw, [height, width, -1])

        label = tf.reshape(tf.cast(img_obj.label, tf.int32), [])
        num_threads = 1
        if shuffle:
            min_after_dequeue = 10000
            capacity = min_after_dequeue + 3 * batch_size
            image_batch, label_batch = tf.train.shuffle_batch([image, label],
                                                              batch_size=batch_size,
                                                              capacity=capacity,
                                                              min_after_dequeue=min_after_dequeue,
                                                              num_threads=num_threads)
        else:
            capacity=10000 + 3 * batch_size
            image_batch, label_batch = tf.train.batch([image, label],
                                                      batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      capacity=capacity)
        return image_batch, label_batch

def load_tfrecord_data(tfrecord_file_path):
    dataset = DataSet(tfrecord_file_path)
    with tf.Session() as sess:
        image_batch, label_batch = dataset.next_batch(64)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images, labels = sess.run((image_batch,label_batch))
        print(images.shape, labels.shape)
        coord.request_stop()
        coord.join(threads)


data_path = "../data/4chars_train.tfrecord"
load_tfrecord_data(data_path)
