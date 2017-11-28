import os

import cv2
import dlib

ECCV_CROP_FACE_DIR = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/face'

detector = dlib.get_frontal_face_detector()


def detect_face(face_filepath):
    if not os.path.exists(ECCV_CROP_FACE_DIR) or not os.path.isdir(ECCV_CROP_FACE_DIR):
        os.makedirs(ECCV_CROP_FACE_DIR)

    img = cv2.imread(face_filepath)
    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))

    if len(dets) > 0:
        for i, d in enumerate(dets):
            print("Left: {} Top: {} Right: {} Bottom: {} score: {} face_type:{}".format(d.left(), d.top(), d.right(),
                                                                                        d.bottom(), scores[i], idx[i]))

            face = img[d.left(): d.right(), d.top(): d.bottom(), :]
            face = cv2.resize(face, (128, 128))
            cv2.imwrite(os.path.join(ECCV_CROP_FACE_DIR, face_filepath.split('/')[-1]), face)
    else:
        h, w, c = img.shape
        face = img[0: w, 0: w, :]
        face = cv2.resize(face, (128, 128))
        cv2.imwrite(os.path.join(ECCV_CROP_FACE_DIR, face_filepath.split('/')[-1]), face)


if __name__ == '__main__':
    eccv_face_dir = "/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/hotornot_face"

    for _ in os.listdir(eccv_face_dir):
        detect_face(os.path.join(eccv_face_dir, _))
