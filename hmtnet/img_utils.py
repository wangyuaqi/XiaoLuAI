import os
import sys

import cv2

sys.path.append('../')
from hmtnet.cfg import cfg


def det_landmarks(image_path):
    """
    detect faces in one image, return face bbox and landmarks
    :param image_path:
    :return:
    """
    import dlib
    predictor = dlib.shape_predictor(cfg['dlib_model'])
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


def crop_faces(img_dir):
    """
    crop face region and show image window
    :param img_dir:
    :return:
    """
    for img_file in os.listdir(img_dir):
        res = det_landmarks(os.path.join(img_dir, img_file))
        for i in range(len(res)):
            bbox = res[i]['bbox']
            image = cv2.imread(os.path.join(img_dir, img_file))
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow('image', image)
            cv2.waitKey()
            cv2.destroyAllWindows()
