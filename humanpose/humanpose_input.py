import os
import tensorflow as tf

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGE_CHANNEL = 3


def read_humanpose(file_queue):
    class HumanPose():
        pass

    label_bytes = 10
    humanPose = HumanPose()
    humanPose.height = IMAGE_HEIGHT
    humanPose.width = IMAGE_WIDTH
    humanPose.depth = IMAGE_CHANNEL

    tf.FixedLengthRecordReader()
