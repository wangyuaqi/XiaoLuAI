import cv2
import numpy as np

from facescore.face_beauty_regressor import det_landmarks


def face_align(face_image_path):
    img = cv2.imread(face_image_path)
    result = det_landmarks(face_image_path)

    face = result.get(0)

    left_eye_left_ldmk = face['landmarks'][36]
    left_eye_right_ldmk = face['landmarks'][39]

    right_eye_left_ldmk = face['landmarks'][42]
    right_eye_right_ldmk = face['landmarks'][45]

    left_center = [int((left_eye_left_ldmk[0] + left_eye_right_ldmk[0]) / 2),
                   int((left_eye_left_ldmk[1] + left_eye_right_ldmk[1]) / 2)]

    right_center = [int((right_eye_left_ldmk[0] + right_eye_right_ldmk[0]) / 2),
                    int((right_eye_left_ldmk[1] + right_eye_right_ldmk[1]) / 2)]

    theta = np.degrees(np.arctan((right_center[1] - left_center[1]) / (right_center[0] - left_center[0])))
    print(theta)

    face_bbox = face['bbox']
    # roi = img[face_bbox[0]: face_bbox[2], face_bbox[1]: face_bbox[3], :]

    rows, cols, chs = img.shape
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), theta, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))

    cv2.imshow('image', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    face_path = '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/face/female_18_AEHEKZB_face_3.jpg'
    # face_align(face_path)
    image = cv2.imread(face_path)
    cv2.findContours(image, mode='')