import math
import os
import sys

import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from scipy import spatial


def detect_face(face_img_file):
    """
    detect face with MTCNN and draw face region with OpenCV
    :param face_img:
    :return:
    """
    img = cv2.imread(face_img_file)
    detector = MTCNN()
    result = detector.detect_faces(img)

    if len(result) != 2:
        print('Only a couple of two faces in one image is allowed!')
    else:

        for _ in result:
            cv2.rectangle(img, (_['box'][0], _['box'][1]), (_['box'][0] + _['box'][2], _['box'][1] + _['box'][3]),
                          (0, 0, 255), 2)
            cv2.circle(img, (_['keypoints']['left_eye'][0], _['keypoints']['left_eye'][1]), 2, (255, 245, 0), -1)
            cv2.circle(img, (_['keypoints']['right_eye'][0], _['keypoints']['right_eye'][1]), 2, (255, 245, 0), -1)
            cv2.circle(img, (_['keypoints']['nose'][0], _['keypoints']['nose'][1]), 2, (255, 245, 0), -1)
            cv2.circle(img, (_['keypoints']['mouth_left'][0], _['keypoints']['mouth_left'][1]), 2, (255, 245, 0), -1)
            cv2.circle(img, (_['keypoints']['mouth_right'][0], _['keypoints']['mouth_right'][1]), 2, (255, 245, 0), -1)
            cv2.putText(img, str(round(_['confidence'], 2)), (_['box'][0], _['box'][1] - 3),
                        0, 0.4, (0, 0, 255), 0, cv2.LINE_AA)

        cv2.imshow('res', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        face1 = img[result[0]['box'][0]:result[0]['box'][0] + result[0]['box'][2],
                result[0]['box'][1]:result[0]['box'][1] + result[0]['box'][3]]
        face2 = img[result[1]['box'][0]:result[1]['box'][0] + result[1]['box'][2],
                result[1]['box'][1]:result[1]['box'][1] + result[1]['box'][3]]

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
        from facescore.features import hog_from_cv, LBP_from_cv

        feature1 = np.concatenate(
            (hog_from_cv(cv2.resize(face1, (128, 128))), LBP_from_cv(cv2.resize(face1, (128, 128)))))
        feature2 = np.concatenate(
            (hog_from_cv(cv2.resize(face2, (128, 128))), LBP_from_cv(cv2.resize(face2, (128, 128)))))

        cos_sim = cal_cos_sim(feature1, feature2)
        print('Cosine Similarity = %f' % cos_sim)

        # detect with dlib
        from facescore import face_beauty_regressor
        result = face_beauty_regressor.det_landmarks(face_img_file)
        geo_dis = cal_geo_dis(result[0]['landmarks'], result[1]['landmarks'])
        print('Geo distance = %f' % geo_dis)


def cal_cos_sim(feature1, feature2):
    """
    calculate cosine similarity
    :param feature1:
    :param feature2:
    :return:
    """
    return 1 - spatial.distance.cosine(feature1, feature2)


def cal_geo_dis(face_ldmk_list1, face_ldmk_list2):
    eu_dis = np.linalg.norm(np.array(face_ldmk_list1) - np.array(face_ldmk_list2))

    return 1 / (1 + math.exp(-eu_dis))


if __name__ == '__main__':
    detect_face('./jay.jpg')
