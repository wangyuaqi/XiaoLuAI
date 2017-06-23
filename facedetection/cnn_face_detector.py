import os

import tensorflow as tf
import cv2

LFW = '/home/lucasx/Documents/Dataset/ImageDataSet/lfw/'
CASCPath = '../res/haarcascade_frontalface_default.xml'
EYE_CASCADE_PATH = '../res/haarcascade_eye.xml'
OPENCV_CROPPED_FACE_DIR = './face/'


def opencv_detect_face(image_filepath):
    """
    run opencv face detector and save the cropped face image
    :param image_filepath:
    :return:
    """
    faceCascade = cv2.CascadeClassifier(CASCPath)
    image = cv2.imread(image_filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.imwrite(os.path.join(OPENCV_CROPPED_FACE_DIR, image_filepath.split('/')[-1]), image[x: x + w, y: y + h])


def face_input():
    # filename_queue = tf.train.string_input_producer(
    #     [os.path.join(OPENCV_CROPPED_FACE_DIR, _) for _ in os.listdir(OPENCV_CROPPED_FACE_DIR)])
    filename_queue = tf.train.string_input_producer(
        ['/home/lucasx/Documents/Dataset/ImageDataSet/lfw/Abraham_Foxman/Abraham_Foxman_0001.jpg'])
    reader = tf.TFRecordReader()
    key, record = reader.read(filename_queue)
    features = tf.parse_single_example(
        record,
        # Defaults are not specified since both keys are required.
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    print(image.shape)


if __name__ == '__main__':
    """
    if tf.gfile.Exists(OPENCV_CROPPED_FACE_DIR):
        tf.gfile.DeleteRecursively(OPENCV_CROPPED_FACE_DIR)
    tf.gfile.MakeDirs(OPENCV_CROPPED_FACE_DIR)
    for person_dir in os.listdir(LFW):
        for each_face in os.listdir(os.path.join(LFW, person_dir)):
            opencv_detect_face(os.path.join(LFW, person_dir, each_face))
    """
    face_input()
