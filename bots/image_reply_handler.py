"""
    unfinished yet!
"""

import cv2
import numpy as np
import matplotlib as plt


def sim_of_images(img1_vector, img2_vector):
    print(np.corrcoef(img1_vector, img2_vector)[0, 1])


def img2vector(image_filepath):
    image = cv2.imread(image_filepath)
    print(image.shape)

    surf = cv2.xfeatures2d.SURF_create(400)
    kp, des = surf.detectAndCompute(image, None)
    img2 = cv2.drawKeypoints(image, kp, None, (255, 0, 0), 4)
    plt.imshow(img2), plt.show()


if __name__ == '__main__':
    img2vector('/home/lucasx/Documents/Dataset/ImageDataSet/Beauty-Faces/2016302110043.jpg')
