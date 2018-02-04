"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: train.py
@time: 2018/2/3 12:46
"""

import os
import tensorflow as tf
from dataset import DataSet
from model_v2 import Model

def train(train_data_path,
          test_data_path,
          val_data_path,
          batch_size=64,
          learning_rate=1e-4,
          drop_rate=0.75,
          regularization_scale=0.0,
          max_steps=1000,
          logdir="../log",
          model_dir="../model"):
    train_data = DataSet(train_data_path)
    test_data = DataSet(test_data_path)
    val_data = DataSet(val_data_path)
    with tf.name_scope("input"):
        x = tf.placeholder(tf.float32, shape=[None, 60, 160, 1], name='x')
        digit1 = tf.placeholder(tf.int32, shape=[None, 1], name='digit1')
        digit2 = tf.placeholder(tf.int32, shape=[None, 1], name='digit2')
        digit3 = tf.placeholder(tf.int32, shape=[None, 1], name='digit3')
        digit4 = tf.placeholder(tf.int32, shape=[None, 1], name='digit4')
        digits_group = [digit1, digit2, digit3, digit4]
        digits_labels = (tf.one_hot(digit1, 10),
                         tf.one_hot(digit2, 10),
                         tf.one_hot(digit3, 10),
                         tf.one_hot(digit4, 10))
    drop_rate_tensor = tf.placeholder(tf.float32)
    digit_logits = Model.inference(x, drop_rate_tensor, tf.contrib.layers.l2_regularizer(regularization_scale))
    with tf.name_scope("softmax"):
        digits_probs = [tf.nn.softmax(logits) for logits in digit_logits]
        pred_digits_tensor = [tf.cast(tf.argmax(prob, axis=1), tf.int32) for prob in digits_probs]

    with tf.name_scope("accuracy"):
        tf_equal = tf.equal(tf.stack(pred_digits_tensor, axis=1), tf.squeeze(tf.stack(digits_group, axis=1)))
        tf_equal = tf.equal(tf.reduce_sum(tf.cast(tf_equal, tf.float32), axis=1), 4)
        accuracy = tf.reduce_mean(tf.cast(tf_equal, tf.float32))

    loss = Model.loss(digit_logits, digits_labels)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    tf.summary.scalar("loss", loss)
    merged_summary = tf.summary.merge_all()

    train_image_batch, train_digits_batch = train_data.next_batch(batch_size)
    test_image_batch, test_digits_batch = test_data.next_batch(1000, shuffle=False)
    val_image_batch, val_digits_batch = val_data.next_batch(batch_size, shuffle=False)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(logdir=logdir, graph=sess.graph)
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            print("Restore model from {}".format(ckpt_state.model_checkpoint_path))
            saver.restore(sess, ckpt_state.model_checkpoint_path)
        for step in range(max_steps+1):
            images, digits = sess.run([train_image_batch, train_digits_batch])
            _, loss_val, summary = sess.run([train_op, loss, merged_summary], feed_dict={
                x: images,
                digit1: digits[:, 0:1],
                digit2: digits[:, 1:2],
                digit3: digits[:, 2:3],
                digit4: digits[:, 3:4],
                drop_rate_tensor: drop_rate
            })

            writer.add_summary(summary, step)

            if step % 20 == 0:
                val_images, val_digits = sess.run([val_image_batch, val_digits_batch])
                train_acc = sess.run(accuracy, feed_dict={
                    x: images,
                    digit1: digits[:, 0:1],
                    digit2: digits[:, 1:2],
                    digit3: digits[:, 2:3],
                    digit4: digits[:, 3:4],
                    drop_rate_tensor: 1.0
                })
                val_acc = sess.run(accuracy, feed_dict={
                    x: val_images,
                    digit1: val_digits[:, 0:1],
                    digit2: val_digits[:, 1:2],
                    digit3: val_digits[:, 2:3],
                    digit4: val_digits[:, 3:4],
                    drop_rate_tensor: 1.0
                })
                print("step: {}, loss: {}, train acc: {}, val acc: {}".format(step, loss_val, train_acc, val_acc))

            if step % 1000 == 0:
                test_images, test_digits = sess.run([test_image_batch, test_digits_batch])
                test_acc = sess.run(accuracy, feed_dict={
                    x: test_images,
                    digit1: test_digits[:, 0:1],
                    digit2: test_digits[:, 1:2],
                    digit3: test_digits[:, 2:3],
                    digit4: test_digits[:, 3:4],
                    drop_rate_tensor: 1.0
                })
                print("=====> Test accuracy: {}".format(test_acc))
                saver.save(sess=sess,
                           save_path=os.path.join(model_dir, "model.ckpt"),
                           global_step=step)

        saver.save(sess=sess,
                   save_path=os.path.join(model_dir, "latest_model.ckpt"))
        writer.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    train_data_path = "../data/4chars_train.tfrecord"
    test_data_path = "../data/4chars_test.tfrecord"
    val_data_path = "../data/4chars_val.tfrecord"
    batch_size = 64
    learning_rate = 1e-4
    drop_rate = 0.7
    regularization_scale = 1e-4
    max_steps = 10000

    train(train_data_path=train_data_path,
          test_data_path=test_data_path,
          val_data_path=val_data_path,
          batch_size=batch_size,
          learning_rate=learning_rate,
          drop_rate=drop_rate,
          regularization_scale=regularization_scale,
          max_steps=max_steps)