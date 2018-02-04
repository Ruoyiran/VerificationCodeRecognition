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
    def _conv_layer_block(scope_name,
                          inputs,
                          filters,
                          kernel_size,
                          drop_rate,
                          use_pool=True,
                          regularization=None):
        with tf.variable_scope(scope_name):
            conv = tf.layers.conv2d(inputs, filters=filters, kernel_size=[kernel_size, kernel_size], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            if use_pool:
                activation = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.layers.dropout(activation, rate=drop_rate)
            layer = dropout
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name + "/conv2d/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name + "/conv2d/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization is not None:
                tf.add_to_collection("loss", regularization(kernel))
            return layer

    @staticmethod
    def _full_connected_block(scope_name, inputs, units, regularization=None):
        with tf.variable_scope(scope_name):
            dense = tf.layers.dense(inputs, units, activation=tf.nn.relu)
            kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name+"/dense/kernel")[0]
            bias = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name+"/dense/bias")[0]
            variable_summaries(kernel)
            variable_summaries(bias)
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope_name+'/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))
        return dense

    @staticmethod
    def inference(x, drop_rate, regularization=None, show_net_summary=True):
        hidden1 = Model._conv_layer_block("hidden_layer1", x, 48, 5, drop_rate=drop_rate, regularization=regularization)
        hidden2 = Model._conv_layer_block("hidden_layer2", hidden1, 64, 5, drop_rate=drop_rate,
                                          regularization=regularization)
        hidden3 = Model._conv_layer_block("hidden_layer3", hidden2, 128, 5, drop_rate=drop_rate,
                                          regularization=regularization)
        hidden4 = Model._conv_layer_block("hidden_layer4", hidden3, 160, 5, drop_rate=drop_rate,
                                          regularization=regularization)
        hidden5 = Model._conv_layer_block("hidden_layer5", hidden4, 192, 5, drop_rate=drop_rate,
                                          regularization=regularization)
        flatten = tf.layers.flatten(hidden5, name="flatten")
        hidden6 = Model._full_connected_block("hidden_layer6", flatten, 1920, regularization=regularization)
        hidden7 = Model._full_connected_block("hidden_layer7", hidden6, 1920, regularization=regularization)
        fc_output = hidden7
        if show_net_summary:
            print(hidden1.get_shape())
            print(hidden2.get_shape())
            print(hidden3.get_shape())
            print(hidden4.get_shape())
            print(hidden5.get_shape())
            print(hidden6.get_shape())
            print(hidden7.get_shape())

        with tf.variable_scope('digit1'):
            dense = tf.layers.dense(fc_output, 10)
            digit1 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit1/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit2'):
            dense = tf.layers.dense(fc_output, 10)
            digit2 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit2/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit3'):
            dense = tf.layers.dense(fc_output, 10)
            digit3 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit3/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))

        with tf.variable_scope('digit4'):
            dense = tf.layers.dense(fc_output, 10)
            digit4 = dense
            if regularization:
                kernel = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'digit4/dense/kernel')[0]
                tf.add_to_collection('loss', regularization(kernel))
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