"""
calculate average face of HZAU masters
"""
import os

import numpy as np
from PIL import Image

BASE_DIR = 'D:/DataSet/hzau'


def avg_face(college='coi', shape=[192, 144, 3]):
    """
    calculate the average face by cascading all face images
    :param college:
    :param shape:
    :return:
    :Verson:1.0
    """
    sum_face = np.zeros(shape)
    for _ in os.listdir(os.path.join(BASE_DIR, college)):
        im = Image.open(os.path.join(BASE_DIR, college, _))
        sum_face = np.add(sum_face, im)

    return sum_face / len(os.listdir(os.path.join(BASE_DIR, college)))


if __name__ == '__main__':
    # avg_face = avg_face('plant')
    # im = Image.fromarray(np.uint8(avg_face))
    # im.show()
    dir = 'E:\DataSet\Face\SCUT-FBP\Faces'
    sum_face = np.zeros([128, 128, 3])
    for _ in os.listdir(dir):
        im = Image.open(os.path.join(dir, _))
        sum_face = np.add(sum_face, im)

    avg_face = sum_face / len(os.listdir(dir))
    im = Image.fromarray(np.uint8(avg_face))
    im.show()
