import cv2
import scipy.io as sio
import numpy as np


def mat_to_image(mat_filepath):
    """
    convert a mat file which contains an image's RGB gray value to JPG image files
    :param mat_filepath:
    :return:
    """
    mat_data = sio.loadmat(mat_filepath)
    data = np.array(mat_data['X'], dtype=np.float32)
    data = np.transpose(data, (3, 0, 1, 2))
    for _ in range(len(data)):
        cv2.imwrite('/tmp/svhnimage/%d.jpg' % _, data[_, :, :, :])
        print('/tmp/svhnimage/%d.jpg has been written successfully~' % _)


if __name__ == '__main__':
    mat_to_image('/home/lucasx/Documents/Dataset/ImageDataSet/SVHN/test_32x32.mat')
