"""
@version: 1.0
@author: Roy
@contact: iranpeng@gmail.com
@file: predict_verification_ode_image.py
@time: 2018/2/4 14:09
"""
import os
import tensorflow as tf
import numpy as np
from model import Model
from captcha.image import ImageCaptcha
from PIL import Image


def generate_image(number_str, output_path):
    image = ImageCaptcha()
    image.write(number_str, output_path)

def predict(image_path, model_path):
    true_code, _ = os.path.splitext(os.path.basename(image_path))
    image = Image.open(image_path)
    image = image.convert("L")

    image_np = np.array(image)
    image_np = np.expand_dims(image_np, axis=2)

    tf.reset_default_graph()
    with tf.name_scope("Input"):
        x = tf.placeholder(tf.float32, shape=[None, 60, 160, 1], name='x')
    digit_logits = Model.inference(x, 1.0, None, show_net_summary=False)
    with tf.name_scope("Softmax"):
        digits_probs = [tf.nn.softmax(logits) for logits in digit_logits]
        pred_digits_tensor = [tf.cast(tf.argmax(prob, axis=1), tf.int32) for prob in digits_probs]

    with tf.name_scope("Predict"):
        predict_tensor = tf.stack(pred_digits_tensor, axis=1)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        print("Restore model from {}".format(model_path))
        saver.restore(sess, model_path)

        predict = sess.run(predict_tensor, feed_dict={
            x: [image_np]
        })
        digits = [str(d) for d in predict[0]]
        predict_code = "".join(digits)
        coord.request_stop()
        coord.join(threads)

        print("Image: {}\nPredict: {}\nActual: {}".format(image_path, predict_code, true_code))

if __name__ == '__main__':
    image_path = "../data/4chars/8426.png"
    model_path = "../model/latest_model.ckpt"
    predict(image_path, model_path)

    tmp_dir = "../tmp"
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)

    max_value = 10000
    for i in range(10):
        number = np.random.randint(0, max_value)
        number_str = "%04d" % number
        output_path = os.path.join(tmp_dir, number_str + ".png")
        generate_image(number_str, output_path)
        predict(output_path, model_path)

