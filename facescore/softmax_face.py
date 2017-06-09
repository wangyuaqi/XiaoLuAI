"""
This demo is the face score detection with softmax regression.
Authorized b LucasX.
Regretfully, its accuracy is only approximately 50%, so we should try deep learning method instead of sofxmax regression!
"""
import os

import tensorflow as tf
from util.image_util import generate_train_and_test_data_bin
import numpy as np

IMAGE_SIZE = 64
CLASS_NUM = 5
CHANNEL_NUM = 3  # 3 for RGB and 1 for gray scale
TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'
BATCH_SIZE = 50
MODEL_CKPT_DIR = "/tmp/face/softmax-face"


def main():
    generate_train_and_test_data_bin()

    training_set = unpickle_bin_to_dict(TRAINING_DATA)
    test_set = unpickle_bin_to_dict(TEST_DATA)
    training_image_nums = len(training_set['labels'])

    W = tf.Variable(tf.zeros([IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM, CLASS_NUM]), name='W')
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM])
    b = tf.Variable(tf.zeros([CLASS_NUM]), name='b')
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUM])
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
    session = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    for _ in range(2000):
        x_batch, y_batch = np.array(training_set['data']).reshape(
            [training_image_nums, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM]), one_hot_encoding(
            np.array(training_set['labels']).reshape(training_image_nums)).transpose()
        session.run(train_step, feed_dict={x: x_batch, y_: y_batch})
        if (_ + 1) % 100 == 0:
            saver.save(session, MODEL_CKPT_DIR)
        print('start train %dth epoch...' % _)
    writer = tf.summary.FileWriter('./graphs', session.graph)
    writer.close()

    test_image_nums = len(test_set['labels'])
    x_test, y_test = np.array(test_set['data']).reshape(
        [test_image_nums, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM]), one_hot_encoding(
        np.array(test_set['labels']).reshape(test_image_nums)).transpose()

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy, feed_dict={x: x_test, y_: y_test}))


def unpickle_bin_to_dict(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


def one_hot_encoding(labels):
    one_hot_labels = np.zeros([len(labels), CLASS_NUM], dtype=np.int32)
    for i in range(len(labels)):
        np.transpose(one_hot_labels)[labels[i]][i] = 1

    return np.transpose(one_hot_labels)


def run_inference_on_image(image):
    import cv2
    img = cv2.imread(image).reshape([1, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM])
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('/tmp/face/softmax-face.meta')
        saver.restore(sess, tf.train.latest_checkpoint('/tmp/face'))
        graph = tf.get_default_graph()
        W = graph.get_tensor_by_name("W:0")
        b = graph.get_tensor_by_name("b:0")
        y = tf.matmul(np.array(img, dtype=np.float32), W) + b
        print(np.max(sess.run(y)))


if __name__ == '__main__':
    # main()
    run_inference_on_image('/tmp/face/test_set/2016315110034.jpg')
