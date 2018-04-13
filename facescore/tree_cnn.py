"""
Tree CNN for Facial Beauty Prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf

# import cv2
from facescore import vgg_face_beauty_regressor
from facescore.config import config


def load_data():
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')

    filelist_df = pd.DataFrame([config['face_image_filename'].format(_) for _ in df['Image']])

    shuffled_indices = np.random.permutation(500)
    test_set_size = 100
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    train_filenames = [filelist_df.iloc[train_indices].tolist()]
    train_labels = filelist_df.iloc[train_indices]
    test_filenames = [filelist_df.iloc[test_indices].tolist()]
    test_labels = df['Attractiveness label'].iloc[test_indices]

    return train_filenames, train_labels, test_filenames, test_labels


def train_input_fn(features, labels, batch_size=32):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()


def tree_cnn(features, labels, mode, params):
    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, 1, activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def train():
    train_filenames, train_labels, test_filenames, test_labels = load_data()

    feature_columns = vgg_face_beauty_regressor.extract_feature()

    model = tf.estimator.Estimator(
        model_fn=tree_cnn,
        params={
            'feature_columns': feature_columns,
            # Two hidden layers of 10 nodes each.
            'hidden_units': [10, 10],
            # The model must choose between 3 classes.
            'n_classes': 1,
        })

    # Train the Model.
    model.train(input_fn=lambda: train_input_fn(train_x, train_labels, batch_size=32), steps=200)

    # Evaluate the model.
    eval_result = model.evaluate(input_fn=lambda: train_input_fn(test_x, train_labels, batch_size=32))

    # The evaluation returns a Python dictionary. The "average_loss" key holds the
    # Mean Squared Error (MSE).
    average_loss = eval_result["average_loss"]

    print("\n" + 80 * "*")
    print("\nRMS error for the test set: ${:.0f}".format(1000 * average_loss ** 0.5))


if __name__ == '__main__':
    tf.app.run(train())
