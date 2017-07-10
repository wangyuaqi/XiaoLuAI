import os
import time

import tensorflow as tf
import scipy.io as sio
import numpy as np

CLASS_NUM = 10
BATCH_SIZE = 128
TRAINING_DATA = '/home/lucasx/Documents/Dataset/ImageDataSet/SVHN/train_32x32.mat'
TEST_DATA = '/home/lucasx/Documents/Dataset/ImageDataSet/SVHN/test_32x32.mat'
MODEL_CKPT_DIR = '/tmp/model/svhn'
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANNEL = 3
TRAINING_NUM = 73257
TEST_NUM = 26032


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=5e-2)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def one_hot_encoding(labels):
    """
    one-hot encoding the label list
    :param labels:
    :return:
    """
    one_hot_labels = np.zeros([len(labels), CLASS_NUM], dtype=np.float32)
    for i in range(len(labels)):
        if labels[i][0] != 10:
            np.transpose(one_hot_labels)[labels[i][0]][i] = 1
        else:
            np.transpose(one_hot_labels)[0][i] = 1

    return one_hot_labels


def load_data(mat_file):
    """
    load the mat file and return the data as well as label
    :param mat_file: matlab file which contains images' rgb gray value and correspondent value
    :return: data and its label
    """
    data = sio.loadmat(mat_file)
    return np.array(data['X'], dtype=np.float32), np.array(data['y'], dtype=np.int64)


def main():
    training_data, training_label = load_data(TRAINING_DATA)
    validation_data, validation_label = tf.transpose(training_data[:, :, :, 70000: TRAINING_NUM],
                                                     perm=(3, 0, 1, 2)), one_hot_encoding(
        training_label[70000: TRAINING_NUM])
    training_data, training_label = tf.transpose(training_data[:, :, :, 0: 70000], perm=(3, 0, 1, 2)), one_hot_encoding(
        training_label[0: 70000])
    test_data, test_label = load_data(TEST_DATA)
    test_data, test_label = tf.transpose(test_data, perm=(3, 0, 1, 2)), one_hot_encoding(test_label)

    x = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNEL], name='Data')
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUM], name='Label')

    W_conv1 = weight_variable([5, 5, IMAGE_CHANNEL, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable([5, 5, 64, 128])
    b_conv3 = bias_variable([128])
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)

    W_conv4 = weight_variable([5, 5, 128, 256])
    b_conv4 = weight_variable([256])
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)

    W_conv5 = weight_variable([5, 5, 256, 64])
    b_conv5 = weight_variable([64])
    h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)
    h_pool5 = max_pool_2x2(h_conv5)

    W_fc1 = weight_variable([64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool1_flat = tf.reshape(h_pool5, [-1, 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 256])
    b_fc2 = bias_variable([256])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    W_fc3 = weight_variable([256, CLASS_NUM])
    b_fc3 = weight_variable([CLASS_NUM])
    y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    if tf.gfile.Exists(os.path.abspath(os.path.join(MODEL_CKPT_DIR, os.pardir))):
        tf.gfile.DeleteRecursively(os.path.abspath(os.path.join(MODEL_CKPT_DIR, os.pardir)))
    tf.gfile.MakeDirs(os.path.abspath(os.path.join(MODEL_CKPT_DIR, os.pardir)))

    with tf.Session() as sess:
        writer = tf.summary.FileWriter('/tmp/model/graphs', sess.graph)
        tf.global_variables_initializer().run()
        tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        current_batch_num = 0
        print('start processing %dth batch...' % (current_batch_num / BATCH_SIZE))
        for i in range(20000):
            offset = (i * BATCH_SIZE) % (training_label.shape[0] - BATCH_SIZE)
            print('offset is %d ...' % offset)
            batch_data = training_data[offset: offset + BATCH_SIZE, :, :, :]
            batch_data = batch_data - np.mean(batch_data.eval()) / (np.sqrt(np.var(batch_data.eval())))
            batch_labels = training_label[offset: offset + BATCH_SIZE]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_data.eval(),
                    y_: batch_labels,
                    keep_prob: 1.0})
                validation_accuracy = accuracy.eval(feed_dict={
                    x: validation_data.eval(), y_: validation_label, keep_prob: 1.0
                })
                print(
                    "step %d, training accuracy %g, validation accuracy %g" % (i, train_accuracy, validation_accuracy))
                saver.save(sess, MODEL_CKPT_DIR)
            train_step.run(feed_dict={x: batch_data.eval(), y_: batch_labels, keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(
            feed_dict={x: test_data.eval(), y_: test_label, keep_prob: 1.0}))
        sess.close()
        writer.close()


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    elapse = end_time - start_time
    print('======================================================')
    print(elapse)
    print('======================================================')