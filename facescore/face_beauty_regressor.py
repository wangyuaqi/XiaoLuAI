import math
import os
from pprint import pprint

import cv2
import dlib
import numpy as np
import pandas as pd
import skimage.color
from skimage import io
from skimage.feature import hog, local_binary_pattern, corner_harris
from sklearn import decomposition
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error

from facescore.vgg_face_beauty_regressor import extract_feature

LABEL_EXCEL_PATH = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Rating_Collection/AttractivenessLabel.xlsx'
FACE_IMAGE_FILENAME = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-{0}.jpg'
# FACE_IMAGE_FILENAME = '/media/lucasx/Document/DataSet/Face/SCUT-FBP/Data_Collection/SCUT-FBP-{0}.jpg'
PREDICTOR_PATH = "/home/lucasx/Documents/PretrainedModels/shape_predictor_68_face_landmarks.dat"
TEST_RATIO = 0.1
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


def prepare_data():
    """
    extract HOG features and split it into train and test dataset
    :return:
    :version:1.0
    """
    df = pd.read_excel(LABEL_EXCEL_PATH, 'Sheet1')
    filename_indexs = df['Image']
    attractiveness_scores = df['Attractiveness label']

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * TEST_RATIO)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    trainset_filenames = filename_indexs.iloc[train_indices]
    trainset_label = attractiveness_scores.iloc[train_indices]
    testset_filenames = filename_indexs.iloc[test_indices]
    testset_label = attractiveness_scores.iloc[test_indices]

    # df = pd.read_excel(LABEL_EXCEL_PATH, 'Sheet1')
    # filename_indexs = df['Image'].tolist()
    # attractiveness_scores = df['Attractiveness label'].tolist()
    #
    # trainset_filenames = filename_indexs[0: int(len(filename_indexs) * TRAIN_RATIO)]
    # testset_filenames = filename_indexs[int(len(filename_indexs) * TRAIN_RATIO) + 1: len(filename_indexs)]
    #
    # trainset_label = attractiveness_scores[0: int(len(attractiveness_scores) * TRAIN_RATIO)]
    # testset_label = attractiveness_scores[
    #                 int(len(attractiveness_scores) * TRAIN_RATIO) + 1:int(len(attractiveness_scores))]

    # extract with HOG features
    # train_set_vector = [HOG(FACE_IMAGE_FILENAME.format(_)) for _ in trainset_filenames]
    # test_set_vector = [HOG(FACE_IMAGE_FILENAME.format(_)) for _ in testset_filenames]

    # extract with LBP features
    # train_set_vector = [LBP(FACE_IMAGE_FILENAME.format(_)) for _ in trainset_filenames]
    # test_set_vector = [LBP(FACE_IMAGE_FILENAME.format(_)) for _ in testset_filenames]

    # extract with HARR features
    # train_set_vector = [HARRIS(FACE_IMAGE_FILENAME.format(_)) for _ in trainset_filenames]
    # test_set_vector = [HARRIS(FACE_IMAGE_FILENAME.format(_)) for _ in testset_filenames]

    # extract with Pixel Value features
    # train_set_vector = [RAW(FACE_IMAGE_FILENAME.format(_)) for _ in trainset_filenames]
    # test_set_vector = [RAW(FACE_IMAGE_FILENAME.format(_)) for _ in testset_filenames]

    # extract with VGG Face features
    train_set_vector = [np.concatenate((extract_feature(FACE_IMAGE_FILENAME.format(_), layer_name='conv5_1'),
                                        extract_feature(FACE_IMAGE_FILENAME.format(_), layer_name='conv4_1')), axis=0)
                        for _ in trainset_filenames]
    test_set_vector = [np.concatenate((extract_feature(FACE_IMAGE_FILENAME.format(_), layer_name='conv5_1'),
                                       extract_feature(FACE_IMAGE_FILENAME.format(_), layer_name='conv4_1')), axis=0)
                       for _ in testset_filenames]

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
    img = (img - np.mean(img)) / np.std(img)
    feature = hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')

    return feature


def LBP(img_path):
    """
    extract LBP features
    :param img_path:
    :return:
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = local_binary_pattern(img, P=8, R=0.2)
    # im = Image.fromarray(np.uint8(feature))
    # im.show()

    return feature.reshape(feature.shape[0] * feature.shape[1])


def HARRIS(img_path):
    """
    extract HARR features
    :param img_path:
    :return:
    :Version:1.0
    """
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)
    feature = corner_harris(img, method='k', k=0.05, eps=1e-06, sigma=1)

    return feature.reshape(feature.shape[0] * feature.shape[1])


def RAW(img_path):
    img = io.imread(img_path)
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return img.reshape(img.shape[0] * img.shape[1])


def hog_from_cv(img):
    """
    extract HOG feature from opencv image object
    :param img:
    :return:
    :Version:1.0
    """
    img = skimage.color.rgb2gray(img)
    img = (img - np.mean(img)) / np.std(img)

    return hog(img, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm='L2-Hys')


def det_landmarks(image_path):
    """
    detect faces in one image, return face bbox and landmarks
    :param image_path:
    :return:
    """
    predictor = dlib.shape_predictor(PREDICTOR_PATH)
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


def PCA(feature_matrix):
    """
    PCA algorithm
    :param feature_matrix:
    :return:
    """
    pca = decomposition.PCA(n_components=20)
    pca.fit(feature_matrix)

    return pca.transform(feature_matrix)


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
    # The 1 in the second argument indicates that we should upsample the image 1 time.
    # This will make everything bigger and allow us to detect more faces.
    # dets = detector(img, 1)
    dets, scores, idx = detector.run(img, 1)
    print("Number of faces detected: {}".format(len(dets)))
    for i, d in enumerate(dets):
        print("Left: {} Top: {} Right: {} Bottom: {} score: {} face_type:{}".format(d.left(), d.top(), d.right(),
                                                                                    d.bottom(), scores[i], idx[i]))
        roi = cv2.resize(image[d.top(): d.bottom(), d.left():d.right(), :], (IMAGE_WIDTH, IMAGE_HEIGHT),
                         interpolation=cv2.INTER_CUBIC)
        # feature = hog_from_cv(roi)
        feature = skimage.color.rgb2gray(roi).reshape(IMAGE_WIDTH * IMAGE_HEIGHT)
        attractiveness = br.predict(feature.reshape(-1, feature.shape[0]))

        cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 225), 2)
        cv2.putText(image, str(round(attractiveness[0], 2)), (d.left() + 5, d.top() - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (106, 106, 255), 0, cv2.LINE_AA)

        cv2.imshow('image', image)
        cv2.imwrite('tmp.png', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def train_model(train_set, test_set, train_label, test_label):
    """
    train ML model and serialize it into a binary pickle file
    :param train_set:
    :param test_set:
    :param train_label:
    :param test_label:
    :return:
    :Version:1.0
    """
    # reg = linear_model.RidgeCV(alphas=[_ * 0.1 for _ in range(1, 1000, 1)])
    reg = linear_model.BayesianRidge()
    # reg = svm.SVR()
    reg.fit(train_set, train_label)
    mae_lr = round(mean_absolute_error(test_label, reg.predict(test_set)), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(test_label, reg.predict(test_set))), 4)
    pc = round(np.corrcoef(test_label, reg.predict(test_set))[0, 1], 4)
    # roc_auc_lr = roc_auc_score(test_label, lr.predict(test_set))
    print('===============The Mean Absolute Error of Linear Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Linear Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Linear Model is {0}===================='.format(pc))

    joblib.dump(reg, './bayes_ridge_regressor.pkl')


def eccv_train_and_test_set(split_csv_filepath):
    """
    split train and test eccv dataset
    :param split_csv_filepath:
    :return:
    :Version:1.0
    """
    df = pd.read_csv(split_csv_filepath)
    filenames = [os.path.join(os.path.dirname(split_csv_filepath), 'hotornot_face', _.replace('.bmp', '.jpg')) for _ in
                 df.iloc[:, 0].tolist()]
    scores = df.iloc[:, 1].tolist()
    flags = df.iloc[:, 2].tolist()

    train_set = dict()
    test_set = dict()

    for i in range(len(flags)):
        if flags[i] == 'train':
            train_set[filenames[i]] = scores[i]
        else:
            test_set[filenames[i]] = scores[i]

    return train_set, test_set


def train_and_eval_eccv(train, test):
    for k, v in train.items():
        train_vec = np.array([LBP(k)])
        train_label = np.array([v])

    for k, v in test.items():
        test_vec = np.array([LBP(k)])
        test_label = np.array([v])

    reg = linear_model.BayesianRidge()
    reg.fit(train_vec, train_label)
    mae_lr = round(mean_absolute_error(test_label, reg.predict(test_vec)), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(test_label, reg.predict(test_vec))), 4)
    pc = round(np.corrcoef(test_label, reg.predict(test_vec))[0, 1], 4)

    print('===============The Mean Absolute Error of Linear Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Linear Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Linear Model is {0}===================='.format(pc))


if __name__ == '__main__':
    # pprint(det_landmarks('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-48.jpg'))
    # train_set, test_set = eccv_train_and_test_set(
    #     '/media/lucasx/Document/DataSet/Face/eccv2010_beauty_data_v1.0/eccv2010_beauty_data/eccv2010_split1.csv')
    # train_and_eval_eccv(train_set, test_set)

    # detect_face_and_cal_beauty('/home/lucasx/ll.jpg')
    train_set_vector, test_set_vector, trainset_label, testset_label = prepare_data()
    train_model(train_set_vector, test_set_vector, trainset_label, testset_label)
    # lbp = LBP('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-48.jpg')
    # hog = HOG('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')  # 512-d
    # harr = HARRIS('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')
