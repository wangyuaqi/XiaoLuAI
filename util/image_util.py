import csv
import logging
import os
import pickle
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from util import config

import cv2
import numpy as np
import tensorflow as tf

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
IMAGE_DEPTH = 3

TRAING_IMAGE_DIR = '/tmp/face/training_set/'
TEST_IMAGE_DIR = '/tmp/face/test_set/'
SCORE_CSV_FILE = '/tmp/face/cvlh_hzau_face.csv'
PICKLE_BIN_DIR = '/tmp/face/face_bin/'

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s  \t', level=logging.DEBUG)


def _raw_image_to_dict(image_dir, id_and_score):
    images_dict = dict()
    images_dict['batch_label'] = 'training batch 1 of 1 of HZAU 2016'
    data_list = []
    label_list = []
    filename_list = []
    for each_image in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, each_image))
        b, g, r = cv2.split(image)
        rgb = np.concatenate((r.reshape((IMAGE_HEIGHT * IMAGE_WIDTH)), g.reshape((IMAGE_HEIGHT * IMAGE_WIDTH)),
                              b.reshape((IMAGE_HEIGHT * IMAGE_WIDTH))),
                             axis=0).reshape(IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_DEPTH)
        label = label_by_range(float(id_and_score[each_image.split(".")[0]]))['label']
        arr = rgb.tolist()
        # data_list.append(arr)
        # data_list.append((arr - np.min(arr)) / (np.max(arr) - np.min(arr)))
        data_list.append((arr - np.mean(arr)) / np.sqrt(np.var(arr)))
        label_list.append(label)
        filename_list.append(each_image)
    images_dict['data'] = np.array(data_list, dtype=np.float32)
    images_dict['labels'] = label_list
    images_dict['filenames'] = filename_list

    return images_dict


def out_hzau_face_metafile():
    content = 'ugly\rmedium down\rmedium\rmedium up\rbeautiful'
    with open(PICKLE_BIN_DIR + 'batch.meta', mode='wt') as f:
        f.write(content)
        f.flush()
        f.close()
    logging.debug('meta data has been generated successfully~')


def label_by_range(score):
    if 0 <= score <= 4.5:
        return {'name': 'ugly', 'label': 0}
    elif 4.6 <= score <= 5.7:
        return {'name': 'medium_down', 'label': 1}
    elif 5.8 <= score <= 6.4:
        return {'name': 'medium', 'label': 2}
    elif 6.5 <= score <= 7.4:
        return {'name': 'medium_up', 'label': 3}
    else:
        return {'name': 'beautiful', 'label': 4}


def _get_id_and_labels_from_csv_score_file(csv_score_file):
    id_and_score = dict()
    with open(csv_score_file) as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            id_and_score[row[0]] = row[2]

    return id_and_score


def unpickle_bin_to_dict(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def pickle_dict_to_bin(dict_, dir_, filename):
    mkdirs_if_dir_not_exists(dir_)
    with open(os.path.join(dir_, filename), 'wb') as f:
        pickle.dump(dict_, f)
    logging.debug('all images have been pickled done!')


def mkdirs_if_dir_not_exists(dir_):
    if not os.path.exists(dir_) or not os.path.isdir(dir_):
        os.makedirs(dir_)


def generate_train_and_test_data_bin():
    extract(config.SOURCE_ZIP_FILE)
    id_and_score = _get_id_and_labels_from_csv_score_file(SCORE_CSV_FILE)
    dict_train = _raw_image_to_dict(TRAING_IMAGE_DIR, id_and_score)
    pickle_dict_to_bin(dict_train, PICKLE_BIN_DIR, 'training_set.bin')
    dict_test = _raw_image_to_dict(TEST_IMAGE_DIR, id_and_score)
    pickle_dict_to_bin(dict_test, PICKLE_BIN_DIR, 'test_set.bin')
    out_hzau_face_metafile()


def crop_images(image_dir, new_size_width, new_size_height, out_dir):
    mkdirs_if_dir_not_exists(out_dir)
    for image in os.listdir(image_dir):
        filepath = os.path.join(image_dir, image)
        res = cv2.resize(cv2.imread(filepath), (new_size_width, new_size_height), interpolation=cv2.INTER_AREA)
        out_path = os.path.join(out_dir, image)
        cv2.imwrite(out_path, res)


def extract(zip_filepath):
    extracted_dir = '/tmp/face/'
    if not tf.gfile.Exists(extracted_dir):
        tf.gfile.MakeDirs(extracted_dir)

    import zipfile
    zip_ref = zipfile.ZipFile(zip_filepath, mode='r')
    zip_ref.extractall(extracted_dir)
    zip_ref.close()


def resize_images(image_dir, save_dir):
    for image in os.listdir(image_dir):
        print(os.path.join(image_dir, image))
        img = cv2.imread(os.path.join(image_dir, image))
        res = cv2.resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(save_dir, image), res)


def one_hot_encoding(labels, class_num):
    one_hot_labels = np.zeros([len(labels), class_num], dtype=np.int32)
    for i in range(len(labels)):
        np.transpose(one_hot_labels)[labels[i]][i] = 1

    return np.transpose(one_hot_labels)


if __name__ == '__main__':
    # generate_train_and_test_data_bin()
    # resize_images('/tmp/face/training_set/', '/tmp/face/training_set/')
    print(unpickle_bin_to_dict('/tmp/face/face_bin/test_set.bin')['data'].shape)
