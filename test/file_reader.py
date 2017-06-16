import os
import tensorflow as tf

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
IMAGE_CHANNEL = 3


def read_bulk_images(image_dir):
    filename_queue = tf.train.string_input_producer([os.path.join(image_dir, _) for _ in os.listdir(image_dir)])
    reader = tf.FixedLengthRecordReader(record_bytes=IMAGE_HEIGHT * IMAGE_WIDTH * IMAGE_CHANNEL)
    print(reader.read(filename_queue))


if __name__ == '__main__':
    # read_bulk_images('/home/lucasx/Documents/Dataset/ImageDataSet/Beauty-Faces')
    arr = tf.Variable([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]], dtype=tf.float32)
    with tf.Session() as session:
        tf.global_variables_initializer().run()

        print(arr.eval())
