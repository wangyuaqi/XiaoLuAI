import tensorflow as tf
from PIL import Image

from facescore import vggface


def main(face_filepath):
    img = Image.open(face_filepath, mode='r')
    img = img.resize((224, 224), Image.ANTIALIAS)
    vggFace = vggface.VGG_FACE_16({'data': img})
    with tf.Session() as sess:
        # Load the data
        vggFace.load('/home/lucasx/ModelZoo/vggface.npy', sess)
        # Forward pass
        output = sess.run(vggFace.get_output(), ...)
        print(output)


if __name__ == '__main__':
    main('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-2.jpg')
