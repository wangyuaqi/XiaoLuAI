import tensorflow as tf
import numpy as np
from util.image_util import generate_train_and_test_data_bin, unpickle_bin_to_dict

IMAGE_SIZE = 64
CLASS_NUM = 5
CHANNEL_NUM = 3  # 3 for RGB and 1 for gray scale
TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'
BATCH_SIZE = 128
TRAINING_SIZE = 766
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
    test_data = unpickle_bin_to_dict(TEST_DATA)['data'].reshape([TEST_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM])
    test_labels = one_hot_encoding(unpickle_bin_to_dict(TEST_DATA)['labels'])

    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM], name='TrainX')
    y_ = tf.placeholder(tf.float32, [None, CLASS_NUM])

    W_conv1 = weight_variable([4, 4, 3, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([16 * 16 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 16 * 16 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 5])
    b_fc2 = bias_variable([5])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    if tf.gfile.Exists(MODEL_CKPT_DIR):
        tf.gfile.DeleteRecursively(MODEL_CKPT_DIR)
    tf.gfile.MakeDirs(MODEL_CKPT_DIR)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        current_batch_num = 0
        print('start processing %dth batch...' % (current_batch_num / BATCH_SIZE))
        for i in range(2000):
            offset = (i * BATCH_SIZE) % (train_labels.shape[0] - BATCH_SIZE)
            batch_data = train_data[offset: offset + BATCH_SIZE]
            batch_labels = train_labels[offset: offset + BATCH_SIZE]
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={
                    x: batch_data, y_: batch_labels, keep_prob: 1.0})
                print("step %d, training accuracy %g" % (i, train_accuracy))
                saver.save(sess, MODEL_CKPT_DIR)
            train_step.run(feed_dict={x: batch_data, y_: batch_labels, keep_prob: 0.5})

        print("test accuracy %g" % accuracy.eval(
            feed_dict={x: test_data, y_: test_labels, keep_prob: 1.0}))
        sess.close()


if __name__ == '__main__':
    main()
