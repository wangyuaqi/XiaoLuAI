import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os

from skimage import io
from skimage.transform import resize

"""
feature extractor
"""
import numpy as np
import skimage.color
from skimage import io
from skimage.feature import hog, local_binary_pattern, corner_harris


def HOG(img_path):
    """
    extract HOG feature
    :param img_path:
    :return:
    :version: 1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')

    return feature


def LBP(img_path):
    """
    extract LBP features
    :param img_path:
    :return:
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = local_binary_pattern(img, P=8, R=0.2)
    # im = Image.fromarray(np.uint8(feature))
    # im.show()

    return feature.reshape(feature.shape[0] * feature.shape[1])


def LBP_from_cv(img):
    """
    extract LBP features from opencv region
    :param img:
    :return:
    """
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = local_binary_pattern(img, P=8, R=0.2)
    # im = Image.fromarray(np.uint8(feature))
    # im.show()

    return feature.reshape(feature.shape[0] * feature.shape[1])


def HARRIS(img_path):
    """
    extract HARR features
    :param img_path:
    :return:
    :Version:1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = corner_harris(img, method='k', k=0.05, eps=1e-06, sigma=1)

    return feature.reshape(feature.shape[0] * feature.shape[1])


def RAW(img_path):
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return img.reshape(img.shape[0] * img.shape[1])


def hog_from_cv(img):
    """
    extract HOG feature from opencv image object
    :param img:
    :return:
    :Version:1.0
    """
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')


def load_state_dict(model, fname):
    """
    Set parameters converted from Caffe models authors of VGGFace2 provide.
    See https://www.robots.ox.ac.uk/~vgg/data/vgg_face2/.
    Arguments:
        model: model
        fname: file name of parameters converted from a Caffe model, assuming the file format is Pickle.
    """
    with open(fname, 'rb') as f:
        weights = pickle.load(f, encoding='latin1')

    own_state = model.state_dict()
    for name, param in weights.items():
        if name in own_state:
            try:
                own_state[name].copy_(torch.from_numpy(param))
            except Exception:
                raise RuntimeError(
                    'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                    'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
        else:
            raise KeyError('unexpected key "{}" in state_dict'.format(name))


def extract_vggface2_feature(img_path, resnet50_model_path="E:/ModelZoo/VGGFace2/resnet50_ft_weight.pkl"):
    resnet50 = models.resnet50()

    load_state_dict(resnet50, resnet50_model_path)
    resnet50.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    image = resize(io.imread(img_path), (224, 224), mode='constant')

    image[:, :, 0] -= np.mean(image[:, :, 0])
    image[:, :, 1] -= np.mean(image[:, :, 1])
    image[:, :, 2] -= np.mean(image[:, :, 2])

    image = np.transpose(image, [2, 0, 1])

    input = torch.from_numpy(image).unsqueeze(0).float()

    resnet50 = resnet50.to(device)
    input = input.to(device)

    for k, v in resnet50.modules():
        print(k)


if __name__ == '__main__':
    extract_vggface2_feature('./jay.jpg')
