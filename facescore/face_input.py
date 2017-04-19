from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import os

IMAGE_SIZE = 144
NUM_CLASSES = 5
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000


def read_face(filename_queue):
    class Face():
        pass

    face = Face()
    label_bytes = 1
    face.height = 144
    face.width = 144
    face.depth = 3
    image_bytes = face.height * face.width * face.depth
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    face.key, value = reader.read(filename_queue)

    record_bytes = tf.decode_raw(value, tf.uint8)
    face.label = tf.cast(tf.strided_slice(record_bytes, [0], tf.int32))

    depth_major = tf.reshape(record_bytes, [label_bytes], [label_bytes + image_bytes])
    face.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return face


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
    num_process_threads = 16
    if shuffle:
        images, label_batch = tf.train.shuffle_batch([image, label], batch_size=batch_size,
                                                     num_threads=num_process_threads,
                                                     capacity=min_queue_examples + 3 * batch_size,
                                                     min_after_dequeue=min_queue_examples)
    else:
        images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_process_threads,
                                             capacity=min_queue_examples + 3 * batch_size)
    tf.summary.image('image', images)

    return images, tf.reshape(label_batch, [batch_size])


def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'training_set.bin')]
    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Error to find file ' + f)
    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_face(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # subtract off the mean and divide by the variance of the pixels
    float_image = tf.image.per_image_standardization(reshaped_image)

    # set the shape of tensors
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # ensure that the random shuffling has a good mixing properties
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
    print('Filling queue with %d CIFAR images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)


def inputs(eval_data, data_dir, batch_size):
    if not eval_data:
        filenames = [os.path.join(data_dir, 'training_set.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
    else:
        filenames = [os.path.join(data_dir, 'test_set.bin')]
        num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

    for f in filenames:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file ' + f)

    filename_queue = tf.train.string_input_producer(filenames)
    read_input = read_face(filename_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    height = IMAGE_SIZE
    width = IMAGE_SIZE

    # Subtract off the mean and divide by the variance of the pixels.
    float_image = tf.image.per_image_standardization(reshaped_image)

    # Set the shapes of tensors.
    float_image.set_shape([height, width, 3])
    read_input.label.set_shape([1])

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(num_examples_per_epoch *
                             min_fraction_of_examples_in_queue)

    # Generate a batch of images and labels by building up a queue of examples.
    return _generate_image_and_label_batch(float_image, read_input.label,
                                           min_queue_examples, batch_size,
                                           shuffle=False)
