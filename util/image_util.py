import os
import csv
import pickle

import cv2
import logging

TRAING_IMAGE_DIR = '/home/lucasx/Documents/crop_images/training_set/'
TEST_IMAGE_DIR = '/home/lucasx/Documents/crop_images/test_set/'
SCORE_CSV_FILE = '/home/lucasx/Documents/Dataset/ImageDataSet/cvlh_hzau_face.csv'

logging.basicConfig(format='%(levelname)s:%(asctime)s:%(message)s  \t', level=logging.DEBUG)


def _raw_image_to_list(image_dir, id_and_score):
    """parse an image file into a list object
        Format: <label><rgb channel values>
    """
    v_list = list()
    for each_image in os.listdir(image_dir):
        image = cv2.imread(os.path.join(image_dir, each_image))
        label = label_by_range(float(id_and_score[each_image.split(".")[0]]))['label']
        logging.info(label)
        b, g, r = cv2.split(image)
        line = label + r + g + b
        v_list.append(line)

    return v_list


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


def pickle_list_to_bin(list_, dir_, filename):
    mkdirs_if_dir_not_exists(dir_)
    with open(os.path.join(dir_, filename), 'wb') as f:
        pickle.dump(list_, f)
    logging.debug('all images have been pickled done!')


def mkdirs_if_dir_not_exists(dir_):
    if not os.path.exists(dir_) or not os.path.isdir(dir_):
        os.makedirs(dir_)


if __name__ == '__main__':
    id_and_score = _get_id_and_labels_from_csv_score_file(SCORE_CSV_FILE)
    v_list = _raw_image_to_list(TRAING_IMAGE_DIR, id_and_score)
    pickle_list_to_bin(v_list, '/tmp/face/', 'training_set.bin')
