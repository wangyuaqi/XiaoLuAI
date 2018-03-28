import cv2
from scipy import spatial
from mtcnn.mtcnn import MTCNN


def detect_face(face_img):
    """
    detect face with MTCNN and draw face region with OpenCV
    :param face_img:
    :return:
    """
    img = cv2.imread(face_img)
    detector = MTCNN()
    result = detector.detect_faces(img)

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

    # import facescore.features
    # feature1 = hov_from_cv(img[_['box'][0]:_['box'][0] + _['box'][2], _['box'][1]:_['box'][1] + _['box'][3]])


def cal_cos_sim(feature1, feature2):
    return 1 - spatial.distance.cosine(feature1, feature2)


if __name__ == '__main__':
    detect_face('E:/jay.jpg')
