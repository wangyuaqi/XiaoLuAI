import os

import cv2

SCUT_FBP = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Data_Collection/Data_Collection'
IMAGE_SIZE = 224


def resample_images(dst_dir='/media/lucasx/Document/DataSet/Face/SCUT-FBP/Processed'):
    if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for _ in os.listdir(SCUT_FBP):
        image = cv2.imread(os.path.join(SCUT_FBP, _))
        h, w, c = image.shape
        if h >= w:
            roi = image[0:w, 0:w, :]
        else:
            roi = image[0:h, 0:h, :]

        roi = cv2.resize(roi, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(os.path.join(dst_dir, _), roi)
        print('write image %s' % os.path.join(dst_dir, _))


def resize_images(dst_dir='/media/lucasx/Document/DataSet/Face/SCUT-FBP/Processed'):
    if not os.path.exists(dst_dir) or not os.path.isdir(dst_dir):
        os.makedirs(dst_dir)

    for _ in os.listdir(SCUT_FBP):
        image = cv2.imread(os.path.join(SCUT_FBP, _))
        re_image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        cv2.imwrite(os.path.join(dst_dir, _), re_image)
        print('write image %s' % os.path.join(dst_dir, _))


if __name__ == '__main__':
    resize_images()
