"""
This demo is the face score detector powered by feed forward neural nets
Authorized by LucasX
"""
import tensorflow as tf
import numpy as np

TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'
IMAGE_SIZE = 32
CHANNEL_NUM = 3
CLASS_NUM = 5


def main():
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM)]

    training_set = unpickle_bin_to_dict(TRAINING_DATA)
    test_set = unpickle_bin_to_dict(TEST_DATA)

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[1000, 2000, 3000, 2000, 1000],
                                                n_classes=5,
                                                model_dir="/tmp/dnn_model")
    training_image_nums = len(training_set['labels'])
    test_image_nums = len(test_set['labels'])

    def get_train_inputs():
        x, y = tf.constant(np.array(training_set['data'], dtype=np.float32).reshape(
            [training_image_nums, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM])), tf.constant(
            np.array(training_set['labels'], dtype=np.int))
        return x, y

    def get_test_inputs():
        x, y = tf.constant(np.array(test_set['data'], dtype=np.float32).reshape(
            [test_image_nums, IMAGE_SIZE * IMAGE_SIZE * CHANNEL_NUM])), tf.constant(
            np.array(test_set['labels'], dtype=np.int))
        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=5000)
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


def one_hot_encoding(labels):
    one_hot_labels = np.zeros([len(labels), CLASS_NUM], dtype=np.int32)
    for i in range(len(labels)):
        np.transpose(one_hot_labels)[labels[i]][i] = 1

    return np.transpose(one_hot_labels)


def init_data():
    from util.image_util import generate_train_and_test_data_bin
    generate_train_and_test_data_bin()


def unpickle_bin_to_dict(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict


if __name__ == '__main__':
    # init_data()
    main()
