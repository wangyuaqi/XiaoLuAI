"""
Intro: a face beauty ranker and Christmas Hat Wearing toolkit for WeChat sharing
Author: LucasX
"""
import math
import cv2
import sys
import os

import numpy as np
import dlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from facescore.face_beauty_regressor import det_mat_landmarks, detect_face_and_cal_beauty

detector = dlib.get_frontal_face_detector()


def main(original_face, scored_face, face_score):
    """
    generate image with a christmas red hat as well as a face beauty score
    :param original_face:
    :param scored_face:
    :param face_score:
    :return:
    """
    h, w, c = scored_face.shape
    if w != 449:
        ratio = float(449.0 / w)
        print('The ratio is %f' % ratio)
        original_face = cv2.resize(original_face, (449, int(ratio * h)))
        scored_face = cv2.resize(scored_face, (449, int(ratio * h)))

        h, w, c = scored_face.shape

    qrcode_size = 100
    text_area_height = 100
    result = 255 * np.ones([h + text_area_height, 2 * w, c], dtype=np.uint8)

    result_json = det_mat_landmarks(original_face)
    if len(result) > 0:
        for index, face in result_json.items():
            hat = cv2.imread('./red_hat.png')

            # hat size should be 1/2 of a face
            hat_size = int((result_json[index]['bbox'][2] - result_json[index]['bbox'][0]) * 0.5)
            hat = cv2.resize(hat, (hat_size, hat_size))

            rows, cols, chs = hat.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -45, 1)
            hat = cv2.warpAffine(hat, M, (cols, rows))

            hat_location_in_org_img = original_face[
                                      result_json[index]['bbox'][1] - hat_size: result_json[index]['bbox'][1],
                                      result_json[index]['bbox'][2]: result_json[index]['bbox'][2] + hat_size, :]
    else:
        print('Sorry, no face detected! Try another one please~')

    result[0: h, 0: w, :] = original_face
    result[0: h, w:2 * w, :] = scored_face

    qrcode = cv2.resize(cv2.imread('./qrcode_wechat.jpg'), (qrcode_size, qrcode_size))
    result[h: h + text_area_height, 2 * w - qrcode_size:2 * w, :] = qrcode
    text_area = result[h: h + text_area_height, 0:2 * w - qrcode_size, :]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(text_area, 'Your face beauty score is {0}, surpassing {1}% people in your friends circle.'
                .format(face_score, rank_percentage(face_score)), (int((2 * w - qrcode_size - 760) / 2), 20), font, 0.6,
                (255, 144, 30), 0, cv2.LINE_AA)
    cv2.putText(text_area, 'AI Technology Supported by @LucasX',
                (int((2 * w - qrcode_size - 300) / 2), 90), font, 0.5, (106, 106, 255), 0, cv2.LINE_AA)
    cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()
    cv2.imwrite('./wechat_share.png', result)


def sigmoid(x):
    """
    logistic sigmoid function
    :param x:
    :return:
    """
    return 1 / (1 + math.exp(-x))


def rank_percentage(face_score):
    """
    calculate surpassing percentage
    :param face_score:
    :return:
    """
    if 0 < face_score <= 30:
        return int(face_score + 10 * np.random.ranf(1)[0])
    elif 30 < face_score <= 40:
        return int(face_score + 10 * np.random.ranf(1)[0])
    elif 40 < face_score <= 50:
        return int(face_score + 10 * np.random.ranf(1)[0])
    elif 50 < face_score <= 60:
        return int(face_score + 10 * np.random.ranf(1)[0])
    else:
        return int(90 + 5 * np.random.ranf(1)[0])


if __name__ == '__main__':
    original_image_filepath = './xulu.jpg'
    score, scored_face = detect_face_and_cal_beauty(original_image_filepath)
    original_face = cv2.imread(original_image_filepath)
    main(original_face, scored_face, score)
