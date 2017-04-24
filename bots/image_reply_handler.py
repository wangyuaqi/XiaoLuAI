"""
    unfinished yet!
"""

import cv2
import numpy as np


def sim_of_images(img1_vector, img2_vector):
    print(np.corrcoef(img1_vector, img2_vector)[0, 1])


def img2vector(image_filepath):
    image = cv2.imread(image_filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    print(dst.reshape([144 * 144, 1]))
    dst = cv2.dilate(dst, None)
    # Threshold for an optimal value, it may vary depending on the image.
    image[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('dst', image)
    # if cv2.waitKey(0) & 0xff == 27:
    #     cv2.destroyAllWindows()

    return dst.reshape([144 * 144, 1])


if __name__ == '__main__':
    img_vec1 = img2vector('/home/lucasx/Documents/crop_images/test_set/2016303010013.jpg')
    img_vec2 = img2vector('/home/lucasx/Documents/crop_images/test_set/2016303010013.jpg')
    sim_of_images(img_vec1, img_vec2)
