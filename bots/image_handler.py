"""
image clustering via HOG features and KMeans algorithm
"""
import os
import shutil

import cv2
import scipy.spatial.distance as distance
from skimage.feature import hog
from sklearn import metrics
from sklearn.cluster import KMeans

IMAGE_BASE_DIR = '/home/lucasx/Documents/Dataset/ImageDataSet/hzau'


def HOG(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')
    return feature


def cosine_sim(vec1, vec2):
    return 1 - distance.cosine(vec1, vec2)


def data_preprocessing():
    image_and_vec = {}
    for sub_dir in os.listdir(IMAGE_BASE_DIR):
        for each_image in os.listdir(os.path.join(IMAGE_BASE_DIR, sub_dir)):
            img_ab_path = os.path.join(IMAGE_BASE_DIR, sub_dir, each_image)
            image_and_vec[img_ab_path] = HOG(img_ab_path)

    return image_and_vec


def main(image_and_vec_dict):
    shutil.rmtree('/tmp/face/')
    flist = list()
    X = list()
    for k, v in image_and_vec_dict.items():
        flist.append(k)
        X.append(v)
    km = KMeans(n_clusters=10, random_state=1).fit(X)
    labels = km.labels_.tolist()
    metrics.silhouette_score(X, labels, metric='euclidean')
    for i in range(len(flist)):
        if not os.path.exists('/tmp/face/%d' % labels[i]) or not os.path.isdir('/tmp/face/%d' % labels[i]):
            os.makedirs('/tmp/face/%d' % labels[i])
        shutil.copyfile(flist[i], '/tmp/face/%d/%s' % (labels[i], flist[i].split('/')[-1]))
    print('all images have been processed done!!!')


if __name__ == '__main__':
    image_and_vec = data_preprocessing()
    main(image_and_vec)
