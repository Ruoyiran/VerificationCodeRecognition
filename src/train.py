"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: train.py
@time: 2018/2/3 12:46
"""

import tensorflow as tf
from dataset import DataSet
from model import Model

def load_tfrecord_data(tfrecord_file_path):
    with tf.name_scope("input_layer"):
        x = tf.placeholder(tf.float32, [None, 60, 160, 3], name="x")
        digit1 = tf.placeholder(tf.int32, [None, 10], name="digit1")
        digit2 = tf.placeholder(tf.int32, [None, 10], name="digit2")
        digit3 = tf.placeholder(tf.int32, [None, 10], name="digit3")
        digit4 = tf.placeholder(tf.int32, [None, 10], name="digit4")
        digits_labels = tf.stack([digit1, digit2, digit3, digit4], axis=1)

    dataset = DataSet(tfrecord_file_path)
    digit_logits = Model.inference(x, 0.7, tf.contrib.layers.l2_regularizer(1e-4))
    loss = Model.loss(digit_logits, digits_labels)
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)
    print(digit_logits.get_shape())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        image_batch, label_batch = dataset.next_batch(1)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        images, labels = sess.run((image_batch, label_batch))
        print(labels)
        coord.request_stop()
        coord.join(threads)

data_path = "../data/4chars_train.tfrecord"
load_tfrecord_data(data_path)
