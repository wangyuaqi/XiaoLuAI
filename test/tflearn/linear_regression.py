import numpy as np
import tensorflow as tf

TRAINING_DATA = 'iris_training.csv'
TEST_DATA = 'iris_test.csv'


def main():
    feature_columns = [tf.contrib.layers.real_valued_column("", dimension=1)]

    training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TRAINING_DATA, target_dtype=np.int,
                                                                       features_dtype=np.float32)
    test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=TEST_DATA, target_dtype=np.int,
                                                                   features_dtype=np.float32)

    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns, hidden_units=[10, 20, 10], n_classes=3,
                                                model_dir="/tmp/dnn_model")

    def get_train_inputs():
        x = tf.constant(training_set.data)
        y = tf.constant(training_set.target)
        return x, y

    def get_test_inputs():
        x = tf.constant(test_set.data)
        y = tf.constant(test_set.target)
        return x, y

    classifier.fit(input_fn=get_train_inputs, steps=2000)
    accuracy_score = classifier.evaluate(input_fn=get_test_inputs,
                                         steps=1)["accuracy"]

    print("\nTest Accuracy: {0:f}\n".format(accuracy_score))


if __name__ == '__main__':
    main()
