"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: train.py
@time: 2018/2/3 12:46
"""

import tensorflow as tf
import numpy as np
from dataset import DataSet
from model import Model

def train(tfrecord_file_path,
          batch_size=64,
          learning_rate=1e-4,
          drop_rate=0.75,
          regularization_scale=0.0,
          max_steps=1000,
          logdir="../log"):
    dataset = DataSet(tfrecord_file_path)
    with tf.name_scope("Input"):
        image_batch, digits_batch = dataset.next_batch(batch_size)
    digit_logits = Model.inference(image_batch, drop_rate, tf.contrib.layers.l2_regularizer(regularization_scale))
    with tf.name_scope("softmax"):
        prediction = tf.nn.softmax(digit_logits)
    with tf.name_scope("accuracy"):
        tf_equal = tf.equal(tf.cast(tf.argmax(prediction, axis=2), tf.int32), digits_batch)
        accuracy = tf.reduce_sum(tf.cast(tf_equal, tf.int32), axis=1)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(accuracy, 4), tf.float32))

    loss = Model.loss(digit_logits, digits_batch)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.summary.scalar("loss", loss)
    merged_summary = tf.summary.merge_all()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(max_steps):
            _, loss_val, summary = sess.run([train_op, loss, merged_summary])
            print("step: {}, loss: {}".format(step, loss_val))
            writer.add_summary(summary, step)
            if step % 20 == 0:
                acc = sess.run(accuracy)
                print("===> Train accuracy: {}".format(acc))
        writer.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    data_path = "../data/4chars_train.tfrecord"
    batch_size = 20
    learning_rate = 1e-3
    drop_rate = 1.0# 0.7
    regularization_scale = 1e-4
    max_steps = 10000

    train(data_path, batch_size, learning_rate, drop_rate, regularization_scale, max_steps=max_steps)