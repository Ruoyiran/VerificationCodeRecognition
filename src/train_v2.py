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
from model_v2 import Model

def train(tfrecord_file_path,
          batch_size=64,
          learning_rate=1e-4,
          drop_rate=0.75,
          regularization_scale=0.0,
          max_steps=1000,
          logdir="../log"):
    dataset = DataSet(tfrecord_file_path)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=[None, 60, 160, 1], name='x')
        digit1 = tf.placeholder(tf.int32, shape=[None, 1], name='digit1')
        digit2 = tf.placeholder(tf.int32, shape=[None, 1], name='digit2')
        digit3 = tf.placeholder(tf.int32, shape=[None, 1], name='digit3')
        digit4 = tf.placeholder(tf.int32, shape=[None, 1], name='digit4')
        digits_labels = (tf.one_hot(digit1, 10),
                         tf.one_hot(digit2, 10),
                         tf.one_hot(digit3, 10),
                         tf.one_hot(digit4, 10))

    digit_logits = Model.inference(x, drop_rate, tf.contrib.layers.l2_regularizer(regularization_scale))
    with tf.name_scope("softmax"):
        # digits_probs = [tf.nn.softmax(logits) for logits in digit_logits]
        # pred_digits = [tf.argmax(prob, axis=1) for prob in digits_probs]
        probs = tf.nn.softmax(digit_logits[0])
        pred_digits = [tf.argmax(probs, axis=1)]

    loss = Model.loss(digit_logits, digits_labels)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.summary.scalar("loss", loss)
    merged_summary = tf.summary.merge_all()

    image_batch, digits_batch = dataset.next_batch(batch_size)
    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for step in range(max_steps):
            images, digits = sess.run([image_batch, digits_batch])
            _, loss_val, summary = sess.run([train_op, loss, merged_summary], feed_dict={
                x: images,
                digit1: digits[:, 0:1],
                digit2: digits[:, 1:2],
                digit3: digits[:, 2:3],
                digit4: digits[:, 3:4],
            })
            print("step: {}, loss: {}".format(step, loss_val))
            writer.add_summary(summary, step)

            if step % 1 == 0:
                pred = sess.run(pred_digits, feed_dict={
                    x: images
                })
                print("pred", pred)

        writer.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    data_path = "../data/4chars_train.tfrecord"
    batch_size = 1
    learning_rate = 1e-4
    drop_rate = 1.0 # 0.7
    regularization_scale = 1e-6
    max_steps = 100

    train(data_path, batch_size, learning_rate, drop_rate, regularization_scale, max_steps=max_steps)