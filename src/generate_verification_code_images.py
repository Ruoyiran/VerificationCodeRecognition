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
np.random.seed(1024)
from captcha.image import ImageCaptcha

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

    sys.stdout.write("\nDone.")
    sys.stdout.flush()

def generate_dataset_tfrecords(data_dir="../data/4chars", test_size=0.1, val_size=0.1, shuffle=True):
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
    assert (len(test_files) == test_count)
    assert (len(val_files) == val_count)
    assert (len(train_files) == train_count)
    print("Total samples: %d" % total_files)
    print("Train samples: %d" % train_count)
    print("Val samples: %d" % val_count)
    print("Test samples: %d" % test_count)


# generate_codes()
generate_dataset_tfrecords()