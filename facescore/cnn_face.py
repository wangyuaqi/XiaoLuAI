import tensorflow as tf
from util.image_util import generate_train_and_test_data_bin, unpickle_bin_to_dict

IMAGE_SIZE = 32
CLASS_NUM = 5
CHANNEL_NUM = 3  # 3 for RGB and 1 for gray scale
TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'
BATCH_SIZE = 50
VALIDATION_SIZE = 100
NUM_EPOCHS = 5000


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


def main():
    generate_train_and_test_data_bin()
    train_data = unpickle_bin_to_dict(TRAINING_DATA)['data']
    train_labels = unpickle_bin_to_dict(TRAINING_DATA)['labels']

    validation_data = train_data[:VALIDATION_SIZE, ...]
    validation_labels = train_labels[:VALIDATION_SIZE]
    train_data = train_data[VALIDATION_SIZE:, ...]
    train_labels = train_labels[VALIDATION_SIZE:]

    test_data = TEST_DATA(TRAINING_DATA)['data']
    test_labels = unpickle_bin_to_dict(TEST_DATA)['labels']
    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]

    train_data_node = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data = tf.placeholder(
        tf.float32,
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL_NUM))

    conv1_weights = tf.Variable(
        tf.truncated_normal([5, 5, CHANNEL_NUM, 32],  # 5x5 filter, depth 32.
                            stddev=0.1,
                            seed=None, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.truncated_normal(
        [5, 5, 32, 64], stddev=0.1,
        seed=None, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))
    fc1_weights = tf.Variable(  # fully connected, depth 512.
        tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512],
                            stddev=0.1,
                            seed=None,
                            dtype=tf.float32))
    fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=tf.float32))
    fc2_weights = tf.Variable(tf.truncated_normal([512, 5],
                                                  stddev=0.1,
                                                  seed=None,
                                                  dtype=tf.float32))
    fc2_biases = tf.Variable(tf.constant(
        0.1, shape=[5], dtype=tf.float32))
