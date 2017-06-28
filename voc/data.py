"""
for Pascal VOC data pre processing, the min size of the image is 96*127*3
"""
import os

import tensorflow as tf
import numpy as np
import cv2

from voc.config import *


def pre_processing(list_):
    for _ in list_:
        print(_)


def get_voc_train_and_val_files_and_labels(imagesets_main_folder):
    """
    get training set and validation set from the given folder, and return the list separately
    :param imagesets_main_folder:
    :return:
    """
    train_set = []
    val_set = []
    for _ in os.listdir(imagesets_main_folder):
        if _.endswith('_train.txt'):
            label = _.split('_')[0].strip()
            with open(os.path.join(imagesets_main_folder, _), mode='rt', encoding='utf-8') as f:
                filelist_of_one_label = [JPG_IMAGES_DIR + _.split(' ')[0].strip() for _ in f.readlines()]
            train_set.append({label: filelist_of_one_label})
        elif _.endswith('_val.txt'):
            label = _.split('_')[0].strip()
            with open(os.path.join(imagesets_main_folder, _), mode='rt', encoding='utf-8') as f:
                filelist_of_one_label = [JPG_IMAGES_DIR + _.split(' ')[0].strip() for _ in f.readlines()]
            val_set.append({label: filelist_of_one_label})
        else:
            pass

    return train_set, val_set


def get_images_shape(image_folder):
    """
    get all images' shape in the given image folder
    :param image_folder:
    :return:
    """
    shape_list = []
    for _ in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder, _))
        if image.shape not in shape_list:
            shape_list.append(image.shape)

    return shape_list


if __name__ == '__main__':
    imageset_main_folder = '/home/lucasx/Documents/Dataset/VOC/VOCtrainval_06-Nov-2007/VOC2007/ImageSets/Main/'
    train_set, val_set = get_voc_train_and_val_files_and_labels(imageset_main_folder)

    list_ = get_images_shape(JPG_IMAGES_DIR)
    print(np.min([_[1] for _ in list_]))
