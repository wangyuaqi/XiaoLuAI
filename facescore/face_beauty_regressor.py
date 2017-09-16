import os

import dlib
import pandas as pd
import skimage.color
from skimage import io
from skimage.feature import hog
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
import cv2

LABEL_EXCEL_PATH = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
FACE_IMAGE_FILENAME = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-{0}.jpg'
TRAIN_RATIO = 0.8
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


def prepare_data():
    """
    extract HOG features and split it into train and test dataset
    :return:
    :version:1.0
    """
    df = pd.read_excel(LABEL_EXCEL_PATH, 'Sheet1')
    filename_indexs = df['Image'].tolist()
    attractiveness_scores = df['Attractiveness label'].tolist()

    trainset_filenames = filename_indexs[0: int(len(filename_indexs) * TRAIN_RATIO)]
    testset_filenames = filename_indexs[int(len(filename_indexs) * TRAIN_RATIO) + 1: len(filename_indexs)]

    trainset_label = attractiveness_scores[0: int(len(attractiveness_scores) * TRAIN_RATIO)]
    testset_label = attractiveness_scores[
                    int(len(attractiveness_scores) * TRAIN_RATIO) + 1:int(len(attractiveness_scores))]

    train_set_vector = [HOG(FACE_IMAGE_FILENAME.format(_)) for _ in trainset_filenames]
    test_set_vector = [HOG(FACE_IMAGE_FILENAME.format(_)) for _ in testset_filenames]

    return train_set_vector, test_set_vector, trainset_label, testset_label


def HOG(img_path):
    """
    extract HOG feature
    :param img_path:
    :return:
    :version: 1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')

    return feature


def hog_from_cv(img):
    """
    extract HOG feature from opencv image object
    :param img:
    :return:
    """
    img = skimage.color.rgb2gray(img)
    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')


def detect_face_and_cal_beauty(face_filepath):
    """
    face detection with dlib
    :param face_filepath:
    :return:
    :version:1.0
    """
    # if the pre-trained model did not exist, then we train it
    if not os.path.exists('./bayes_ridge_regressor.pkl'):
        train_set_vector, test_set_vector, trainset_label, testset_label = prepare_data()
        train_model(train_set_vector, test_set_vector, trainset_label, testset_label)

    br = joblib.load('./bayes_ridge_regressor.pkl')

    image = cv2.imread(face_filepath)
    detector = dlib.get_frontal_face_detector()
    # win = dlib.image_window()
    img = io.imread(face_filepath)
    # The 1 in the second argument indicates that we should upsample the image
    # 1 time.  This will make everything bigger and allow us to detect more
    # faces.
    # dets = detector(img, 1)
    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Left: {} Top: {} Right: {} Bottom: {} score: {} face_type:{}".format(d.left(), d.top(), d.right(),
                                                                                    d.bottom(), scores[i], idx[i]))
        roi = cv2.resize(image[d.top(): d.bottom(), d.left():d.right(), :], (IMAGE_WIDTH, IMAGE_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
        hog = hog_from_cv(roi)
        attractiveness = br.predict(hog.reshape(-1, hog.shape[0]))

        cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 225), 2)
        cv2.putText(image, str(attractiveness[0]), (d.left() + 5, d.top() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (106, 106, 255), 0, cv2.LINE_AA)

        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # win.clear_overlay()
        # win.set_image(img)
        # win.add_overlay(dets)


def train_model(train_set, test_set, train_label, test_label):
    br = linear_model.BayesianRidge()
    br.fit(train_set, train_label)
    mse_lr = mean_squared_error(test_label, br.predict(test_set))
    r2_lr = r2_score(test_label, br.predict(test_set))
    # roc_auc_lr = roc_auc_score(test_label, lr.predict(test_set))
    print('===============The Mean Square Error of Linear Model is {0}===================='.format(mse_lr))
    print('===============The R^2 Value of Linear Model is {0}===================='.format(r2_lr))

    joblib.dump(br, './bayes_ridge_regressor.pkl')


if __name__ == '__main__':
    detect_face_and_cal_beauty('/home/lucasx/faces.jpg')
