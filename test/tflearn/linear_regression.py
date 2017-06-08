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


if __name__ == '__main__':
    main()
