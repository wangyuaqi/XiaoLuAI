# generate TFRecord files for SCUT-FBP dataset

import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

from facescore.config import config


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_images(excel_path=config['label_excel_path']):
    df = pd.read_excel(excel_path, 'Sheet1')
    img_indices = df['Image'].tolist()
    data = []
    label = df['Attractiveness label']
    for img_index in img_indices:
        data.append(cv2.imread(config['face_image_filename'].format(img_index)))

    data_dict = {'data': data, 'label': label}

    return data_dict


def convert_to_tfrecord(output_file):
    """Converts a file to TFRecords."""
    print('Generating %s' % output_file)
    with tf.python_io.TFRecordWriter(output_file) as record_writer:
        data_dict = read_images()

        data = data_dict['data']
        labels = data_dict['label']
        num_entries_in_batch = len(labels)
        for i in range(num_entries_in_batch):
            example = tf.train.Example(features=tf.train.Features(
                feature={
                    'image': _bytes_feature(data[i].tobytes()),
                    'label': _int64_feature(int(labels[i]))
                }))
            record_writer.write(example.SerializeToString())


def main():
    if not os.path.exists('./tf_records'):
        os.makedirs('./tf_records')

    output_file = os.path.join('./tf_records/SCUT-FBP.tfrecords')
    try:
        os.remove(output_file)
    except OSError:
        pass
    # Convert to tf.train.Example and write the to TFRecords.
    convert_to_tfrecord(output_file)
    print('Done!')


if __name__ == '__main__':
    main()
