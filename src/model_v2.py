"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: model.py
@time: 2018/2/3 12:29
"""

import tensorflow as tf

def variable_summaries(var):
    with tf.name_scope("summaries"):
        mean = tf.reduce_mean(var)
        stddev = tf.reduce_mean(tf.square((var - mean)))
        tf.summary.scalar("mean", mean)
        tf.summary.scalar("stddev", stddev)
        tf.summary.scalar("max", tf.reduce_max(var))
        tf.summary.scalar("min", tf.reduce_min(var))
        tf.summary.histogram("histogram", var)

class Model(object):

    @staticmethod
    def inference(x, drop_rate, regularization=None):
        with tf.variable_scope("hidden_layer1"):
            conv = tf.layers.conv2d(x, filters=48, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden1 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer1/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer1/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization is not None:
                tf.add_to_collection("loss", regularization(kernel))

        with tf.variable_scope('hidden_layer2'):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden2 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer2/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer2/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer3'):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden3 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer3/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer3/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer4'):
            conv = tf.layers.conv2d(hidden3, filters=160, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden4 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer4/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer4/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer5'):
            conv = tf.layers.conv2d(hidden4, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden5 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer5/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer5/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer5/conv2d/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer6'):
            conv = tf.layers.conv2d(hidden5, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden6 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer6/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer6/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer6/conv2d/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer7'):
            conv = tf.layers.conv2d(hidden6, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden7 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer7/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer7/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer7/conv2d/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer8'):
            conv = tf.layers.conv2d(hidden7, filters=192, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=1, padding='same')
            dropout = tf.layers.dropout(pool, rate=drop_rate)
            hidden8 = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer8/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer8/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer8/conv2d/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        flatten = tf.layers.flatten(hidden8, name="flatten")

        with tf.variable_scope('hidden_layer9'):
            dense = tf.layers.dense(flatten, 3072, activation=tf.nn.relu)
            hidden9 = dense
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer9/dense/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer9/dense/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer9/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('hidden_layer10'):
            dense = tf.layers.dense(hidden9, 3072, activation=tf.nn.relu)
            hidden10 = dense
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer10/dense/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, "hidden_layer10/dense/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'hidden_layer10/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(hidden10, 10)
            digit1 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit1/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(hidden10, 10)
            digit2 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit2/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(hidden10, 10)
            digit3 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit3/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(hidden10, 10)
            digit4 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit4/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))
        print(hidden1.get_shape())
        print(hidden2.get_shape())
        print(hidden3.get_shape())
        print(hidden4.get_shape())
        print(hidden5.get_shape())
        print(hidden6.get_shape())
        print(hidden7.get_shape())
        print(hidden8.get_shape())
        print(hidden9.get_shape())
        print(hidden10.get_shape())

        digits_logits = (digit1, digit2, digit3, digit4)
        return digits_logits

    @staticmethod
    def loss(digits_logits, digits_labels):
        with tf.name_scope("loss"):
            with tf.name_scope("digit1_loss"):
                digit1_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=digits_logits[0], labels=digits_labels[0]))
            with tf.name_scope("digit2_loss"):
                digit2_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=digits_logits[1], labels=digits_labels[1]))
            with tf.name_scope("digit3_loss"):
                digit3_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=digits_logits[2], labels=digits_labels[2]))
            with tf.name_scope("digit4_loss"):
                digit4_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(logits=digits_logits[3], labels=digits_labels[3]))
            total_loss = digit1_loss + digit2_loss + digit3_loss + digit4_loss
            loss_collection = tf.get_collection("loss")
            if loss_collection:
                total_loss += tf.add_n(loss_collection)

        return total_loss