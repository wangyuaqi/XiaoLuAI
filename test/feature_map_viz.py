import os
import sys

import numpy as np
import cv2
import tensorflow as tf
from scipy.misc import imread, imresize

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from facescore.vgg_face import vgg_face
from facescore.config import config


def extract_feature(image_filepath, layer_name='conv5_1'):
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)
        # output = extract_deep_feature(VGG_FACE_MODEL_MAT_FILE, input_maps)

    # read sample image
    img = imread(image_filepath, mode='RGB')
    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})
        feature = out[layer_name]
        sess.close()

    return feature


def extract_conv_feature(img, layer_name='conv5_1'):
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(config['vgg_face_model_mat_file'], input_maps)
        # output = extract_deep_feature(VGG_FACE_MODEL_MAT_FILE, input_maps)

    img = (img - img.mean()) / img.std()
    img = imresize(img, [224, 224])

    # run the graph
    with tf.Session(graph=graph) as sess:
        [out] = sess.run([output], feed_dict={input_maps: [img]})
        feature = out[layer_name]
        sess.close()

    return feature


def vis_feature_map(fm):
    """
    feature map visualization
    :param data:
    :return:
    """
    fm = (fm - fm.min()) / (fm.max() - fm.min())
    fm = fm[:, :, 0:3]
    fm = cv2.resize(fm, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('fm', fm)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    fm = extract_feature('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-40.jpg')
    vis_feature_map(fm)
