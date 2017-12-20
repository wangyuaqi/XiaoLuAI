import math
import os
import sys

import cv2
import dlib
import numpy as np
import scipy
import pandas as pd
import skimage.color
from skimage import io
from skimage.feature import hog, local_binary_pattern, corner_harris
from sklearn import decomposition
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from facescore.config import *
from facescore.vgg_face_beauty_regressor import extract_feature, extract_conv_feature
from facescore.face_beauty_regressor import PCA


def test_model():
    model = joblib.load('/home/lucasx/PycharmProjects/XiaoLuAI/facescore/model/eccv_fbp_dcnn_bayes_reg.pkl')
    path = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/face/female_18_AEASGUH_face_1.jpg'

    fm = np.concatenate((extract_feature(path, layer_name="conv5_2"), extract_feature(path, layer_name="conv5_3")),
                        axis=0)

    print(fm.shape)
    feature = PCA(fm, config['num_of_components'])

    score = model.fit(feature)
    print(score)


if __name__ == '__main__':
    test_model()
