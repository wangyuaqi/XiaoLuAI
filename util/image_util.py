import csv
import logging
import os
import pickle
import sys
import urllib.request

import cv2
import numpy as np

IMAGE_WIDTH = 144
IMAGE_HEIGHT = 144
IMAGE_DEPTH = 3
TRAING_IMAGE_DIR = '/tmp/face/Documents/face_dataset/training_set/'
TEST_IMAGE_DIR = '/tmp/face/face_dataset/test_set/'
SCORE_CSV_FILE = '/tmp/face/face_dataset/cvlh_hzau_face.csv'
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
        data_list.append(rgb.tolist())
        label_list.append(label)
        filename_list.append(each_image)
    images_dict['data'] = np.array(data_list, dtype=np.uint8)
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


def maybe_download_and_extract():
    """Download and extract the zip file"""
    data_url = 'https://github.com/EclipseXuLu/XiaoLuAI/blob/master/res/face_dataset.zip'
    dest_directory = '/tmp/face/'
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = data_url.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
                                                             float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(data_url, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filepath, statinfo.st_size, 'bytes.')
    extracted_dir_path = os.path.join(dest_directory, 'face_bin')
    if not os.path.exists(extracted_dir_path):
        import zipfile
        zip_ref = zipfile.ZipFile(filepath, 'r')
        zip_ref.extractall(dest_directory)
        zip_ref.close()


if __name__ == '__main__':
    maybe_download_and_extract()
    # print(unpickle_bin_to_dict('/tmp/face/training_set.bin'))
    # print(unpickle_bin_to_dict('/home/lucasx/Documents/cifar-10-batches-py/data_batch_1'))
    # _generate_train_and_test_data_bin()
    # unpickle_bin_to_dict('/tmp/face/test_set.bin')
