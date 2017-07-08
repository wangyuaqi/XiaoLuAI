import os

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors.nearest_centroid import NearestCentroid

CORPUS_DIR = 'E:/NLP/tc-corpus-answer/answer/'
FEATURE_NUM = 2000


def corpus_to_tfidf_vector(corpus_list):
    vectorizer = CountVectorizer(min_df=1, max_features=FEATURE_NUM)
    transformer = TfidfTransformer(smooth_idf=False)
    X = vectorizer.fit_transform(corpus_list)
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray()


def get_label_and_tfidf(corpus_dir):
    corpus_list = []
    for each_txt in os.listdir(corpus_dir):
        with open(os.path.join(corpus_dir, each_txt), mode='r', encoding='latin1') as f:
            corpus_list.append(''.join(f.readlines()))

    return {int(each_txt.split('-')[0].replace('C', '')): corpus_to_tfidf_vector(corpus_list)}


def prepare_train_and_test_corpus(tfidf_matrix_with_label_dict_list):
    """
    ERROR!!!
    :param tfidf_matrix_with_label_dict_list:
    :return:
    """
    # X_train = np.array([None, FEATURE_NUM])
    # X_test = np.array([None, FEATURE_NUM])
    # y_train = np.array([None])
    # y_test = np.array([None])

    X_train = list()
    X_test = list()
    y_train = list()
    y_test = list()

    train_sum = int(len(tfidf_matrix_with_label_dict_list) * 0.8)
    test_sum = len(tfidf_matrix_with_label_dict_list) - train_sum

    for tfidf_matrix_with_label_dict in tfidf_matrix_with_label_dict_list:
        (label, tfidf_matrix), = tfidf_matrix_with_label_dict.items()
        train_num = int(0.8 * len(tfidf_matrix))
        test_num = len(tfidf_matrix) - train_num
        for _ in tfidf_matrix[0: train_num]:
            X_train.append(_)
        for _ in tfidf_matrix[train_num: len(tfidf_matrix)]:
            X_test.append(_)

        y_train += [label for _ in range(train_num)]
        y_test += [label for _ in range(test_num)]

    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)


def text_classify(X_train, X_test, y_train, y_test):
    print('=' * 100)
    print('start launching MLP Classifier......')
    mlp = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(50, 30, 20, 20, 20, 30, 50), random_state=1)
    mlp.fit(X_train, y_train)
    print('finish launching MLP Classifier, the test accuracy is {:.5%}'.format(mlp.score(X_test, y_test)))

    print('=' * 100)
    print('start launching SVM Classifier......')
    svc = svm.SVC(decision_function_shape='ovo')
    svc.fit(X_train, y_train)
    print('finish launching SVM Classifier, the test accuracy is {:.5%}'.format(svc.score(X_test, y_test)))

    print('=' * 100)
    print('start launching Decision Tree Classifier......')
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    print('finish launching Decision Tree Classifier, the test accuracy is {:.5%}'.format(
        dtree.score(X_test, y_test)))

    print('=' * 100)
    print('start launching KNN Classifier......')
    knn = NearestCentroid()
    knn.fit(X_train, y_train)
    print('finish launching KNN Classifier, the test accuracy is {:.5%}'.format(knn.score(X_test, y_test)))

    print('=' * 100)
    print('start launching Random Forest Classifier......')
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    print('finish launching Random Forest Classifier, the test accuracy is {:.5%}'.format(rf.score(X_test, y_test)))


if __name__ == '__main__':
    label_and_tfidf_list = []
    for _ in os.listdir(CORPUS_DIR):
        corpus_list_and_label = get_label_and_tfidf(os.path.join(CORPUS_DIR, _))
        label_and_tfidf_list.append(corpus_list_and_label)

    X_train, X_test, y_train, y_test = prepare_train_and_test_corpus(label_and_tfidf_list)
    text_classify(X_train, X_test, y_train, y_test)
