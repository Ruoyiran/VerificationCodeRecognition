'''
@version: 1.0
@author: royran
@contact: iranpeng@gmail.com
@file: generate_verification_code_images.py
@time: 2018/2/1 22:17
'''
import os
import sys
import numpy as np
from captcha.image import ImageCaptcha
from tf_utils import TFRecordWriterHelper
from PIL import Image

np.random.seed(1024)

def generate_image(number, output_path):
    image = ImageCaptcha()
    image.write(number, output_path)

def generate_codes(num_chars=4, max_images=10000, img_ext=".png"):
    out_dir = "../data/%dchars" % num_chars
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    num_images = min(10**num_chars, max_images)
    print("Generating {0} images to {1}".format(num_images, out_dir))
    for i in range(num_images):
        sys.stdout.write("\rProcessing {}/{}".format(i+1, num_images))
        sys.stdout.flush()
        number = "%04d" % i
        image_path = os.path.join(out_dir, number + img_ext)
        generate_image(number, image_path)

    sys.stdout.write("\nDone.\n")
    sys.stdout.flush()


def _convert_to_tfrecord(data_dir, file_list, out_dir, tfrecord_file_name, gray_scale=True):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    tfrecord_helper = TFRecordWriterHelper(os.path.join(out_dir, tfrecord_file_name))
    for i, file_name in enumerate(file_list):
        file_path = os.path.join(data_dir, file_name)
        img = Image.open(file_path)
        if gray_scale:
            img = img.convert("L")
        image_np = np.array(img)
        rows = image_np.shape[0]
        cols = image_np.shape[1]
        name, ext = os.path.splitext(os.path.basename(file_name))
        digits = [int(d) for d in list(name)]
        digits = np.asarray(digits)
        sys.stdout.write("\rProcessing {0}/{1}".format(i+1, len(file_list)))
        sys.stdout.flush()
        tfrecord_helper.write_tf_example(height=rows,
                                         width=cols,
                                         digits=digits.tobytes(),
                                         image_raw=image_np.tobytes())

    tfrecord_helper.close()
    sys.stdout.write("\nFinised.\n")
    sys.stdout.flush()


def generate_dataset_tfrecords(data_dir="../data/4chars", out_dir="../data", test_size=0.1, val_size=0.1, shuffle=True):
    if not os.path.exists(data_dir):
        print("{} no such file or director".format(data_dir))
        return
    assert(test_size + val_size < 1.0)
    files = os.listdir(data_dir)
    total_files = len(files)
    test_count = int(total_files*test_size)
    val_count = int(total_files*val_size)
    train_count = total_files - test_count - val_count
    train_files, test_files, val_files = [], [], []
    if shuffle:
        np.random.shuffle(files)
    for i, file_name in enumerate(files):
        if i < test_count:
            test_files.append(file_name)
        elif i >= test_count and i < test_count + val_count:
            val_files.append(file_name)
        else:
            train_files.append(file_name)
    assert (len(train_files) == train_count)
    assert (len(test_files) == test_count)
    assert (len(val_files) == val_count)
    print("Total samples: %d" % total_files)
    print("Train samples: %d" % train_count)
    print("Val samples: %d" % val_count)
    print("Test samples: %d" % test_count)
    _convert_to_tfrecord(data_dir, train_files, out_dir, "4chars_train.tfrecord")
    _convert_to_tfrecord(data_dir, test_files, out_dir, "4chars_test.tfrecord")
    _convert_to_tfrecord(data_dir, val_files, out_dir, "4chars_val.tfrecord")

if __name__ == '__main__':
    generate_codes()
    generate_dataset_tfrecords()