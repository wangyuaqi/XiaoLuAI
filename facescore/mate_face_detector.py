"""
mate face compare
"""
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
        print('Only a couple with two faces in one image is allowed!')
        sys.exit(0)
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
        geo_dis = cal_geo_dis(get_geo_feature(result[0]['landmarks']), get_geo_feature(result[1]['landmarks']))
        print('Geo distance = %f' % geo_dis)

        similarity = 0.8 * cos_sim + (1 - 0.8) * geo_dis

        print('Mate Index is %f ' % similarity)


def get_geo_feature(face_ldmk):
    """
    get geometry feature vector from 68 facial landmarks
    :param face_ldmk:
    :return:
    """
    # dis between left outmost eye and right outmost eye
    d1 = np.linalg.norm(np.array(face_ldmk[37]) - np.array(face_ldmk[26]))

    # width of mouth
    d2 = np.linalg.norm(np.array(face_ldmk[49]) - np.array(face_ldmk[65]))

    # height of mouth
    d3 = np.linalg.norm(np.array(face_ldmk[52]) - np.array(face_ldmk[58]))

    # width of left eye
    d4 = np.linalg.norm(np.array(face_ldmk[43]) - np.array(face_ldmk[46]))

    # height of left eye
    d5 = np.linalg.norm(np.array(face_ldmk[48]) - np.array(face_ldmk[44]))

    # width of right eye
    d6 = np.linalg.norm(np.array(face_ldmk[40]) - np.array(face_ldmk[37]))

    # height of right eye
    d7 = np.linalg.norm(np.array(face_ldmk[42]) - np.array(face_ldmk[38]))

    # dis between eyes' center and nose
    d8 = np.linalg.norm(np.array(face_ldmk[34]) - np.array(face_ldmk[28]))

    # face width
    d9 = np.linalg.norm(np.array(face_ldmk[2]) - np.array(face_ldmk[16]))

    # face height
    d10 = np.linalg.norm(np.array(face_ldmk[9]) - (np.array(face_ldmk[22]) + np.array(face_ldmk[23])) / 2)

    # width of left eyebrow
    d11 = np.linalg.norm(np.array(face_ldmk[18]) - np.array(face_ldmk[22]))

    # width of right eyebrow
    d12 = np.linalg.norm(np.array(face_ldmk[43]) - np.array(face_ldmk[46]))

    return np.array([d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12])


def cal_cos_sim(feature1, feature2):
    """
    calculate cosine similarity
    :param feature1:
    :param feature2:
    :return:
    """
    return 1 - spatial.distance.cosine(feature1, feature2)


def cal_geo_dis(face_ldmk_list1, face_ldmk_list2):
    """
    calculate euclidean distance
    :param face_ldmk_list1:
    :param face_ldmk_list2:
    :return:
    """

    eu_dis = np.linalg.norm(np.array(face_ldmk_list1) - np.array(face_ldmk_list2))

    return 1 / (1 + math.exp(-eu_dis))


if __name__ == '__main__':
    detect_face('./jay.jpg')
