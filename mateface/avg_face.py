"""
calculate average face of HZAU masters
"""
import os

import numpy as np
from PIL import Image

BASE_DIR = 'E:/DataSet/HZAU'


def avg_face(year=2016, college=317, shape=None):
    """
    calculate the average face by cascading all face images
    :param college:
    :param shape:
    :return:
    :Verson:1.0
    """
    if shape is None:
        shape = [192, 144, 3]

    sum_face = np.zeros(shape)
    for _ in os.listdir(os.path.join(BASE_DIR, str(year), str(college))):
        im = Image.open(os.path.join(BASE_DIR, str(year), str(college), _))
        im = im.resize((144, 192), Image.ANTIALIAS)
        sum_face = np.add(sum_face, im)

    return sum_face / len(os.listdir(os.path.join(BASE_DIR, str(year), str(college))))


if __name__ == '__main__':
    avg_face = avg_face()
    im = Image.fromarray(np.uint8(avg_face))
    im.show()
