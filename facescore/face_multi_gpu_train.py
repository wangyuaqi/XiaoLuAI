from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import re
import time
from datetime import datetime

import numpy as np
import tensorflow as tf

from facescore import face
from util.image_util import _generate_train_and_test_data_bin

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/face/',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 5000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('num_gpus', 2,
                            """How many GPUs to use.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")


def tower_loss(scope):
    images, labels = face.distorted_inputs()
    logits = face.inference(images)

    _ = face.loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for i in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]' % face.TOWER_NAME, '', i.op.name)
        tf.summary.scalar(loss_name, i)

    return total_loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grad_and_vars)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def train():
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = (face.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size)
        decay_steps = int(num_batches_per_epoch * face.NUM_EPOCHS_PER_DECAY)

        lr = tf.train.exponential_decay(face.INITIAL_LEARNING_RATE, global_step, decay_steps,
                                        face.LEARNING_RATE_DECAY_FACTOR)
        opt = tf.train.GradientDescentOptimizer(lr)

        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (face.TOWER_NAME, i)) as scope:
                        loss = tower_loss((scope))
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

                # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            face.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation from the last tower summaries.
        summary_op = tf.summary.merge(summaries)

        # Build an initialization operation to run below.
        init = tf.global_variables_initializer()

        # Start running operations on the Graph. allow_soft_placement must be set to
        # True to build towers on GPU, as some of the ops do not have GPU
        # implementations.
        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in range(FLAGS.max_steps):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus

                format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                              'sec/batch)')
                print(format_str % (datetime.now(), step, loss_value,
                                    examples_per_sec, sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    _generate_train_and_test_data_bin()
    train()


if __name__ == '__main__':
    tf.app.run()
