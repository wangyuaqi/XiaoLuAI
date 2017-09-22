import numpy as np
import tensorflow as tf
from scipy.misc import imread, imresize
from scipy.io import loadmat

from facescore.vgg_face import vgg_face

VGG_FACE_MODEL_MAT_FILE = '/home/lucasx/ModelZoo/vgg-face.mat'


def extract_feature(image_filepath, layer_name='fc7'):
    graph = tf.Graph()
    with graph.as_default():
        input_maps = tf.placeholder(tf.float32, [None, 224, 224, 3])
        output, average_image, class_names = vgg_face(VGG_FACE_MODEL_MAT_FILE, input_maps)
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

    return feature.reshape(-1, 1)


if __name__ == '__main__':
    feature = extract_feature(
        '/media/lucasx/02C5-4B45/FaceBeautyPrediction/Tensorflow-VGG-face/Aamir_Khan_March_2015.jpg')

