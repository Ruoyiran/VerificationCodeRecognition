"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: eval.py
@time: 2018/2/4 12:55
"""
import tensorflow as tf
from dataset import DataSet
from model import Model


def eval(data_path, batch_size, model_dir):
    tf.reset_default_graph()
    data = DataSet(data_path)
    image_batch, digits_batch = data.next_batch(batch_size, False)
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=[None, 60, 160, 1], name='x')
        digit1 = tf.placeholder(tf.int32, shape=[None, 1], name='digit1')
        digit2 = tf.placeholder(tf.int32, shape=[None, 1], name='digit2')
        digit3 = tf.placeholder(tf.int32, shape=[None, 1], name='digit3')
        digit4 = tf.placeholder(tf.int32, shape=[None, 1], name='digit4')
        digits_group = [digit1, digit2, digit3, digit4]
    digit_logits = Model.inference(x, 1.0, None)
    with tf.name_scope("Softmax"):
        digits_probs = [tf.nn.softmax(logits) for logits in digit_logits]
        pred_digits_tensor = [tf.cast(tf.argmax(prob, axis=1), tf.int32) for prob in digits_probs]

    with tf.name_scope("Accuracy"):
        tf_equal = tf.equal(tf.stack(pred_digits_tensor, axis=1), tf.squeeze(tf.stack(digits_group, axis=1)))
        tf_equal = tf.equal(tf.reduce_sum(tf.cast(tf_equal, tf.float32), axis=1), 4)
        accuracy = tf.reduce_mean(tf.cast(tf_equal, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ckpt_state = tf.train.get_checkpoint_state(model_dir)
        if ckpt_state and ckpt_state.model_checkpoint_path:
            print("Restore model from {}".format(ckpt_state.model_checkpoint_path))
            saver.restore(sess, ckpt_state.model_checkpoint_path)
        else:
            sess.run(tf.global_variables_initializer())
        images, digits = sess.run([image_batch, digits_batch])
        acc = sess.run(accuracy, feed_dict={
            x: images,
            digit1: digits[:, 0:1],
            digit2: digits[:, 1:2],
            digit3: digits[:, 2:3],
            digit4: digits[:, 3:4],
        })
        coord.request_stop()
        coord.join(threads)
        print("DataSet: %s, Accuracy: %.2f%%" % (data_path, acc * 100))

if __name__ == '__main__':
    train_data_path = "../data/4chars_train.tfrecord"
    test_data_path = "../data/4chars_test.tfrecord"
    val_data_path = "../data/4chars_val.tfrecord"
    model_dir = "../model"
    eval(train_data_path, 1000, model_dir)
    eval(test_data_path, 1000, model_dir)
    eval(val_data_path, 1000, model_dir)