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


def det_landmarks(image_path, dlib_model="E:/ModelZoo/shape_predictor_68_face_landmarks.dat"):
    """
    detect faces in one image, return face bbox and landmarks
    :param dlib_model:
    :param image_path:
    :return:
    """
    if not os.path.exists(dlib_model):
        print('Please download pretrained dlib model from:\n')
        print('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2')
        sys.exit(0)

    import dlib
    predictor = dlib.shape_predictor(dlib_model)
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(image_path)
    faces = detector(img, 1)

    result = {}
    if len(faces) > 0:
        for k, d in enumerate(faces):
            shape = predictor(img, d)
            result[k] = {"bbox": [d.left(), d.top(), d.right(), d.bottom()],
                         "landmarks": [[shape.part(i).x, shape.part(i).y] for i in range(68)]}

    return result


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
        #
        # for _ in result:
        #     cv2.rectangle(img, (_['box'][0], _['box'][1]), (_['box'][0] + _['box'][2], _['box'][1] + _['box'][3]),
        #                   (0, 0, 255), 2)
        #     cv2.circle(img, (_['keypoints']['left_eye'][0], _['keypoints']['left_eye'][1]), 2, (255, 245, 0), -1)
        #     cv2.circle(img, (_['keypoints']['right_eye'][0], _['keypoints']['right_eye'][1]), 2, (255, 245, 0), -1)
        #     cv2.circle(img, (_['keypoints']['nose'][0], _['keypoints']['nose'][1]), 2, (255, 245, 0), -1)
        #     cv2.circle(img, (_['keypoints']['mouth_left'][0], _['keypoints']['mouth_left'][1]), 2, (255, 245, 0), -1)
        #     cv2.circle(img, (_['keypoints']['mouth_right'][0], _['keypoints']['mouth_right'][1]), 2, (255, 245, 0), -1)

        face1 = img[result[0]['box'][0]:result[0]['box'][0] + result[0]['box'][2],
                result[0]['box'][1]:result[0]['box'][1] + result[0]['box'][3]]
        face2 = img[result[1]['box'][0]:result[1]['box'][0] + result[1]['box'][2],
                result[1]['box'][1]:result[1]['box'][1] + result[1]['box'][3]]

        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
        from mateface.features import hog_from_cv, LBP_from_cv

        feature1 = np.concatenate(
            (hog_from_cv(cv2.resize(face1, (128, 128))), LBP_from_cv(cv2.resize(face1, (128, 128)))))
        feature2 = np.concatenate(
            (hog_from_cv(cv2.resize(face2, (128, 128))), LBP_from_cv(cv2.resize(face2, (128, 128)))))

        cos_sim = cal_cos_sim(feature1, feature2)
        print('Appearance Similarity = %f' % cos_sim)

        # detect with dlib
        ldmk = det_landmarks(face_img_file)
        geo_dis = cal_geo_dis(get_geo_feature(ldmk[0]['landmarks']), get_geo_feature(ldmk[1]['landmarks']))
        print('Geo Distance = %f' % geo_dis)

        similarity = 0.8 * cos_sim + (1 - 0.8) * geo_dis

        print('Mate Index is %.2f ' % (similarity * 100))

        text_height = 50
        img_new = 255 * np.ones([img.shape[0] + text_height, img.shape[1], 3], dtype=np.uint8)
        img_new[0:img.shape[0], 0:img.shape[1], :] = img

        cv2.rectangle(img_new, (result[0]['box'][0], result[0]['box'][1]),
                      (result[0]['box'][0] + result[0]['box'][2], result[0]['box'][1] + result[0]['box'][3]),
                      (255, 0, 0), 2)

        cv2.rectangle(img_new, (result[1]['box'][0], result[1]['box'][1]),
                      (result[1]['box'][0] + result[1]['box'][2], result[1]['box'][1] + result[1]['box'][3]),
                      (0, 0, 255), 2)
        cv2.putText(img_new, "The Mate Face Index is %.2f" % similarity,
                    (5, img.shape[0] + 10), 6, 0.4, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('img_new', img_new)
        cv2.waitKey()
        cv2.destroyAllWindows()
        cv2.imwrite('./meteface.jpg', img_new)


def get_geo_feature(face_ldmk):
    """
    get geometry feature vector from 68 facial landmarks
    :param face_ldmk:
    :return:
    """
    # dis between left outmost eye and right outmost eye
    d1 = np.linalg.norm(np.array(face_ldmk[36]) - np.array(face_ldmk[45]))

    # width of mouth
    d2 = np.linalg.norm(np.array(face_ldmk[48]) - np.array(face_ldmk[55]))

    # height of mouth
    d3 = np.linalg.norm(np.array(face_ldmk[57]) - np.array(face_ldmk[51]))

    # width of left eye
    d4 = np.linalg.norm(np.array(face_ldmk[36]) - np.array(face_ldmk[39]))

    # height of left eye
    d5 = np.linalg.norm(np.array(face_ldmk[37]) - np.array(face_ldmk[41]))

    # width of right eye
    d6 = np.linalg.norm(np.array(face_ldmk[42]) - np.array(face_ldmk[45]))

    # height of right eye
    d7 = np.linalg.norm(np.array(face_ldmk[44]) - np.array(face_ldmk[46]))

    # dis between eyes' center and nose
    d8 = np.linalg.norm(np.array(face_ldmk[27]) - np.array(face_ldmk[33]))

    # face width
    d9 = np.linalg.norm(np.array(face_ldmk[1]) - np.array(face_ldmk[15]))

    # face height
    d10 = np.linalg.norm(np.array(face_ldmk[8]) - (np.array(face_ldmk[20]) + np.array(face_ldmk[23])) / 2)

    # width of left eyebrow
    d11 = np.linalg.norm(np.array(face_ldmk[17]) - np.array(face_ldmk[21]))

    # width of right eyebrow
    d12 = np.linalg.norm(np.array(face_ldmk[22]) - np.array(face_ldmk[26]))

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
    detect_face('./lt.jpg')
