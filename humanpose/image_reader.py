import os
import tensorflow as tf


def read_images(image_dir):
    image_filenames = [os.path.join(image_dir, _) for _ in os.listdir(image_dir)]
    filename_queue = tf.train.string_input_producer(image_filenames)
    image_reader = tf.WholeFileReader()
    _, image_file = image_reader.read(filename_queue)
    image = tf.image.decode_image(image_file)
    with tf.Session() as session:
        print(session.run(image))


if __name__ == '__main__':
    read_images('/home/lucasx/Documents/Dataset/ImageDataSet/test')
