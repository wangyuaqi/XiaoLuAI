import os
from pprint import pprint

import cv2
import numpy as np
import pandas as pd

from facescore.face_beauty_regressor import *

if __name__ == '__main__':
    # face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/faceBAK3'
    # csv_file = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split1.csv'
    # train_set, test_set = eccv_train_and_test_set(csv_file)
    #
    # print(len(train_set.keys()))

    size_list = []
    eccv_face_dir = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/hotornot_face'
    for _ in os.listdir(eccv_face_dir):
        img = cv2.imread(os.path.join(eccv_face_dir, _))
        if img.shape not in size_list:
            size_list.append(img.shape)

    print(size_list)
