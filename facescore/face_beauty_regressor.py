import math
import os
import sys

import cv2
import dlib
import numpy as np
import scipy
import pandas as pd
import skimage.color
from skimage import io
from skimage.feature import hog, local_binary_pattern, corner_harris
from sklearn import decomposition
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from facescore.config import *
from facescore.vgg_face_beauty_regressor import extract_feature, extract_conv_feature


def split_train_and_test_data():
    """
    extract facial features and split it into train and test set
    :return:
    :version:1.0
    """
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')
    filename_indexs = df['Image']
    attractiveness_scores = df['Attractiveness label']

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * config['test_ratio'])
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    trainset_filenames = filename_indexs.iloc[train_indices]
    trainset_label = attractiveness_scores.iloc[train_indices]
    testset_filenames = filename_indexs.iloc[test_indices]
    testset_label = attractiveness_scores.iloc[test_indices]

    # extract Deep Features
    train_set_vector = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                                        extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                                       axis=0) for _ in trainset_filenames]
    test_set_vector = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                                       extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                                      axis=0) for _ in testset_filenames]

    return train_set_vector, test_set_vector, trainset_label, testset_label


def prepare_data():
    """
    return the dataset and correspondent labels
    :return:
    """
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')
    filename_indexs = df['Image']
    attractiveness_scores = df['Attractiveness label']

    # extract with Pixel Value features
    dataset = [HOG(config['face_image_filename'].format(_)) for _ in filename_indexs]
    """
    dataset = [np.concatenate((extract_feature(config['face_image_filename'].format(_), layer_name='conv5_1'),
                               extract_feature(config['face_image_filename'].format(_), layer_name='conv4_1')),
                              axis=0) for _ in filename_indexs]
    """

    return dataset, attractiveness_scores


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
    predictor = dlib.shape_predictor(config['predictor_path'])
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


def detect_face_and_cal_beauty(face_filepath='./talor.jpg'):
    """
    face detection with dlib
    :param face_filepath:
    :return:
    :version:1.0
    """
    print('start scoring your face...')
    # if the pre-trained model did not exist, then we train it
    if not os.path.exists(config['reg_model']):
        train_set_vector, test_set_vector, trainset_label, testset_label = prepare_data()
        train_model(train_set_vector, test_set_vector, trainset_label, testset_label)

    br = joblib.load(config['reg_model'])

    result = det_landmarks(face_filepath)

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
        roi = cv2.resize(image[d.top(): d.bottom(), d.left():d.right(), :],
                         (config['image_size'], config['image_size']),
                         interpolation=cv2.INTER_CUBIC)

        feature = np.concatenate(
            (extract_conv_feature(roi, layer_name='conv5_1'), extract_conv_feature(roi, layer_name='conv4_1')), axis=0)
        attractiveness = br.predict(feature.reshape(-1, feature.shape[0]))

        for index, face in result.items():
            # cv2.rectangle(image, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 225), 2)
            cv2.rectangle(image, (face['bbox'][0], face['bbox'][1]), (face['bbox'][2], face['bbox'][3]), (0, 255, 225),
                          2)
            cv2.putText(image, str(round(attractiveness[0] * 20, 2)), (d.left() + 5, d.top() - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (106, 106, 255), 0, cv2.LINE_AA)

            for ldmk in face['landmarks']:
                cv2.circle(image, (ldmk[0], ldmk[1]), 2, (255, 245, 0), -1)

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
    reg = linear_model.BayesianRidge()
    reg.fit(train_set, train_label)

    predicted_label = reg.predict(test_set)
    mae_lr = round(mean_absolute_error(test_label, predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(test_label, predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)
    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))

    if not os.path.exists('./model') or not os.path.isdir('./model'):
        os.makedirs('./model')

    joblib.dump(reg, config['reg_model'])
    print('The regression model has been persisted...')


def cv_train(dataset, labels, cv=10):
    """
    train model with cross validation
    :param model:
    :param dataset:
    :param labels:
    :param cv:
    :return:
    """
    reg = linear_model.BayesianRidge()
    mae_list = -cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='neg_mean_absolute_error')
    rmse_list = np.sqrt(-cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='neg_mean_squared_error'))
    pc_list = cross_val_score(reg, dataset, labels, cv=cv, n_jobs=-1, scoring='r2')

    print(mae_list)
    print(rmse_list)
    print(pc_list)

    print('=========The Mean Absolute Error of Model is {0}========='.format(np.mean(mae_list)))
    print('=========The Root Mean Square Error of Model is {0}========='.format(np.mean(rmse_list)))
    print('=========The Pearson Correlation of Model is {0}========='.format(np.mean(pc_list)))

    if not os.path.exists('./model') or not os.path.isdir('./model'):
        os.makedirs('./model')

    joblib.dump(reg, config['reg_model'])
    print('The regression model has been persisted...')


def eccv_train_and_test_set(split_csv_filepath):
    """
    split train and test eccv dataset
    :param split_csv_filepath:
    :return:
    :Version:1.0
    """
    df = pd.read_csv(split_csv_filepath)
    filenames = [os.path.join(os.path.dirname(split_csv_filepath), 'face', _.replace('.bmp', '.jpg')) for _ in
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
    """
    train and test eccv dataset
    :param train:
    :param test:
    :return:
    """
    train_vec = list()
    train_label = list()
    test_vec = list()
    test_label = list()

    for k, v in train.items():
        train_vec.append(np.concatenate(extract_feature(k, layer_name="conv5_1"), extract_feature(k, layer_name="conv4_1"), axis=0))
        train_label.append(v)

    for k, v in test.items():
        test_vec.append(np.concatenate(extract_feature(k, layer_name="conv5_1"), extract_feature(k, layer_name="conv4_1"), axis=0))
        test_label.append(v)

    reg = linear_model.BayesianRidge()
    reg.fit(np.array(train_vec), np.array(train_label))

    predicted_label = reg.predict(np.array(test_vec))
    mae_lr = round(mean_absolute_error(np.array(test_label), predicted_label), 4)
    rmse_lr = round(math.sqrt(mean_squared_error(np.array(test_label), predicted_label)), 4)
    pc = round(np.corrcoef(test_label, predicted_label)[0, 1], 4)

    print('===============The Mean Absolute Error of Model is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Model is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Model is {0}===================='.format(pc))


if __name__ == '__main__':
    train_set, test_set = eccv_train_and_test_set(config['eccv_dataset_split_csv_file'])
    train_and_eval_eccv(train_set, test_set)

    # dataset, label = prepare_data()
    # cv_train(dataset, label)

    # train_set_vector, test_set_vector, trainset_label, testset_label = split_train_and_test_data()
    # train_model(train_set_vector, test_set_vector, trainset_label, testset_label)

    # detect_face_and_cal_beauty('./talor.jpg')

    # lbp = LBP('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-48.jpg')
    # hog = HOG('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')  # 512-d
    # harr = HARRIS('/media/lucasx/Document/DataSet/Face/SCUT-FBP/Faces/SCUT-FBP-39.jpg')
