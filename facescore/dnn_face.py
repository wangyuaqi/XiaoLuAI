"""
This demo is the face score detector powered by feed forward neural nets
Authorized by LucasX
"""
import tensorflow as tf
import numpy as np

TRAINING_DATA = '/tmp/face/face_bin/training_set.bin'
TEST_DATA = '/tmp/face/face_bin/test_set.bin'


def main():
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

    training_set = unpickle_bin_to_dict(TRAINING_DATA)
    test_set = unpickle_bin_to_dict(TEST_DATA)

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=4,
                                                model_dir="/tmp/dnn_model")

    def get_train_inputs():
        x = tf.constant(np.array(training_set['data']), tf.float32)
        y = tf.constant(np.array(training_set['labels']), tf.int32)
        return x, y

    def get_test_inputs():
        x = tf.constant(np.array(test_set['data']), tf.float32)
        y = tf.constant(np.array(test_set['labels']), tf.int32)
        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=50000)
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


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
