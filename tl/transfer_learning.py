import tensorflow as tf
import numpy as np


def run_inference_on_image(image):
    """Runs inference on an image.

    Args:
      image: Image file name.

    Returns:
      Nothing
    """
    if not tf.gfile.Exists(image):
        tf.logging.fatal('File does not exist %s', image)
    image_data = tf.gfile.FastGFile(image, 'rb').read()

    saver = tf.train.Saver()
    # Creates graph from saved GraphDef.
    with tf.Session() as sess:
        # Some useful tensors:
        # 'softmax:0': A tensor containing the normalized prediction across
        #   1000 labels.
        # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
        #   float description of the image.
        # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
        #   encoding of the image.
        # Runs the softmax tensor by feeding the image_data as input to the graph.
        saver.restore(sess, '/tmp/model')
        softmax_tensor = sess.graph.get_tensor_by_name('fcb2:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)
        print(predictions)


if __name__ == '__main__':
    run_inference_on_image('/home/lucasx/9.jpg')
