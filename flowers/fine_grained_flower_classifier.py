import os

import tensorflow as tf
import scipy.io as sio
import numpy as np

from flowers.config import config


def load_mat_data(mat_file):
    """
    load the mat file
    :param mat_file: matlab file
    :return: data content
    """
    data = sio.loadmat(mat_file)

    return data


def prepare_data():
    """
    data prepare
    :return:
    """
    labels = load_mat_data(config['flower102_labels'])['labels'].ravel()

    dataset = load_mat_data(config['flower102_split'])
    train_ids, val_ids, test_ids = dataset['trnid'][0], dataset['valid'][0], dataset['tstid'][0]

    def fit_filenames(id, prefix='image_0'):
        if 1 <= id < 10:
            return prefix + '000' + str(id) + '.jpg'
        elif 10 <= id < 100:
            return prefix + '00' + str(id) + '.jpg'
        elif 100 <= id < 1000:
            return prefix + '0' + str(id) + '.jpg'
        elif 1000 <= id < 10000:
            return prefix + str(id) + '.jpg'

    train_filenames = [os.path.join(config['flower102_images_dir'], fit_filenames(_)) for _ in train_ids]
    val_filenames = [os.path.join(config['flower102_images_dir'], fit_filenames(_)) for _ in val_ids]
    test_filenames = [os.path.join(config['flower102_images_dir'], fit_filenames(_)) for _ in test_ids]

    train_labels = [labels[_ - 1] for _ in train_ids]
    val_labels = [labels[_ - 1] for _ in val_ids]
    test_labels = [labels[_ - 1] for _ in test_ids]

    return train_filenames, val_filenames, test_filenames, train_labels, val_labels, test_labels


def train():
    pass


if __name__ == '__main__':
    prepare_data()
