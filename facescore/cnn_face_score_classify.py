"""
face scoring with CNN
"""
import os, sys

import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from util.image_util import generate_train_and_test_data_bin, unpickle_bin_to_dict

IMAGE_SIZE = 64
CLASS_NUM = 5
CHANNEL_NUM = 3  # 3 for RGB and 1 for gray scale
TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'
BATCH_SIZE = 64
TRAINING_SIZE = 766
VALIDATION_SIZE = 66
TEST_SIZE = 240
MODEL_CKPT_DIR = "/tmp/face/cnn-face"


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def one_hot_encoding(labels):
    one_hot_labels = np.zeros([len(labels), CLASS_NUM], dtype=np.int32)
    for i in range(len(labels)):
        np.transpose(one_hot_labels)[labels[i]][i] = 1

    return one_hot_labels


def main():
    generate_train_and_test_data_bin()
    train_data = unpickle_bin_to_dict(TRAINING_DATA)['data'].reshape(
        [TRAINING_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])
    train_labels = one_hot_encoding(unpickle_bin_to_dict(TRAINING_DATA)['labels'])

    # train_data = tf.image.random_brightness(train_data, 50)
    validation_data = train_data[-1 - VALIDATION_SIZE: -1]
    validation_labels = train_labels[-1 - VALIDATION_SIZE: -1]

    train_data = train_data[0: TRAINING_SIZE - VALIDATION_SIZE]
    train_labels = train_labels[0: TRAINING_SIZE - VALIDATION_SIZE]

    test_data = unpickle_bin_to_dict(TEST_DATA)['data'].reshape([TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])
    test_labels = one_hot_encoding(unpickle_bin_to_dict(TEST_DATA)['labels'])

    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM], name='TrainX')
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUM])

    W_conv1 = weight_variable([4, 4, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([4, 4, 64, 128])
    b_conv2 = bias_variable([128])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([4, 4, 128, 256])
    b_conv3 = bias_variable([256])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_fc1 = weight_variable([8 * 8 * 256, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool3, [-1, 8 * 8 * 256])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # L2 regularization for the fully connected parameters to avoid over-fitting.
    regularizers = (tf.nn.l2_loss(W_fc1) + tf.nn.l2_loss(b_fc1) +
                    tf.nn.l2_loss(W_fc2) + tf.nn.l2_loss(b_fc2))

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # Add the regularization term to the loss.
    cross_entropy += 1e-4 * regularizers

    global_step = tf.placeholder(tf.float32, name='GlobalStep')
    learning_rate = tf.train.exponential_decay(
        0.01,  # Base learning rate.
        global_step,  # Current index into the dataset.
        1000,  # Decay step.
        0.95,  # Decay rate.
        staircase=True)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        current_batch_num = 0
        print('start processing %dth batch...' % (current_batch_num / BATCH_SIZE))
        for i in range(10000):
            offset = (i * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            batch_data = train_data[offset: offset + BATCH_SIZE]
            batch_labels = train_labels[offset: offset + BATCH_SIZE]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_data, y_: batch_labels, keep_prob: 1.0})
                validation_accuracy = accuracy.eval(feed_dict={
                    x: validation_data, y_: validation_labels, keep_prob: 1.0
                })
                print(
                    "step %d, training accuracy %g, validation accuracy %g" % (i, train_accuracy, validation_accuracy))
                saver.save(sess, MODEL_CKPT_DIR)
            train_step.run(feed_dict={x: batch_data, y_: batch_labels, global_step: i, keep_prob: 0.5})
            # print(learning_rate.eval())

        print("test accuracy %g" % accuracy.eval(
            feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))
        sess.close()


if __name__ == '__main__':
    main()
