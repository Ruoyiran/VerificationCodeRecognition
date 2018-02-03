'''
@version: 1.0
@author: royran
@contact: iranpeng@gmail.com
@file: tf_utils.py
@time: 2018/2/2 13:07
'''
import os
import atexit
import tensorflow as tf

"""
@reference: https://github.com/tensorflow/tensorflow/blob/r1.5/tensorflow/core/example/feature.proto
"""
class TfExampleDecoder(object):

    def bytes_feature(self, value):
        if isinstance(value, list):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def int64_feature(self, value):
        if isinstance(value, list):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def float_feature(self, value):
        if isinstance(value, list):
            return tf.train.Feature(float_list=tf.train.FloatList(value=value))
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


class TFRecordWriterHelper(object):
    def __init__(self, path):
        print("Creating tfrecord file: {}".format(path))
        atexit.register(self.close)
        self.writer = tf.python_io.TFRecordWriter(path)
        self.tfExampleDecoder = TfExampleDecoder()

    def write_tf_example(self, **feature_map):
        if self.writer is None:
            return

        features = self._get_features(**feature_map)
        example = tf.train.Example(features=tf.train.Features(feature=features))
        self.writer.write(example.SerializeToString())

    def _get_features(self, **feature_map):
        features = {}
        for key in feature_map:
            feature = feature_map[key]
            tf_feature = None
            if isinstance(feature, int):
                tf_feature = self.tfExampleDecoder.int64_feature(feature)
            elif isinstance(feature, float):
                tf_feature = self.tfExampleDecoder.float_feature(feature)
            elif isinstance(feature, bytes):
                tf_feature = self.tfExampleDecoder.bytes_feature(feature)
            else:
                print("WARNING: unknown feature type: {} for key: '{}', [int float bytes] type expected.".format(type(feature), key))
            if tf_feature is not None and key not in features:
                features[key] = tf_feature
        return features

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None


class ImageObject(object):
    def __init__(self):
        self.image_raw = tf.Variable([], dtype=tf.string)
        self.height = tf.Variable([], dtype=tf.int64)
        self.width = tf.Variable([], dtype=tf.int64)
        self.digits = tf.Variable([], dtype=tf.string)


class TfRecordReaderHelper(object):

    def load_image_object(self, tfrecord_file_path):
        if not os.path.exists(tfrecord_file_path):
            print("'{}' no such file or directory.".format(tfrecord_file_path))
            return None
        filename_queue = tf.train.string_input_producer([tfrecord_file_path])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(serialized_example, features={
            "height": tf.FixedLenFeature([], tf.int64),
            "width": tf.FixedLenFeature([], tf.int64),
            "digits": tf.FixedLenFeature([], tf.string),
            "image_raw": tf.FixedLenFeature([], tf.string),
        })
        image_obj = ImageObject()
        image_obj.height = features["height"]
        image_obj.width = features["width"]
        image_obj.digits = features["digits"]
        image_obj.image_raw = features["image_raw"]
        return image_obj