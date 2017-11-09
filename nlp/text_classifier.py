import os

import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neural_network import MLPClassifier

CORPUS_DIR = '/home/lucasx/Documents/Dataset/FudanNLPCorpus'
USER_DICT = './user_dict.txt'
STOP_WORDS = './stopwords.txt'
FEATURE_NUM = 2000
TEST_RATIO = 0.2


def corpus_to_tfidf_vector(corpus_list):
    """
    convert segmented corpus in a list into TF-IDF array
    :param corpus_list:
    :return:
    """
    vectorizer = CountVectorizer(min_df=1, max_features=FEATURE_NUM)
    transformer = TfidfTransformer(smooth_idf=False)
    X = vectorizer.fit_transform(corpus_list)
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray()


def prepare_data():
    """
    vectorize the corpora and split them into train set and test set
    :return:
    """
    jieba.load_userdict(USER_DICT)
    jieba.analyse.set_stop_words(STOP_WORDS)

    datasets = []
    labels = []

    for each_type in os.listdir(CORPUS_DIR):
        for each_txt in os.listdir(os.path.join(CORPUS_DIR, each_type)):
            labels.append(int(each_txt.split('-')[0].replace('C', '')))
            with open(os.path.join(CORPUS_DIR, each_type, each_txt), mode='r', encoding='latin1') as f:
                content = ''.join(f.readlines())
                datasets.append(' '.join(jieba.cut(content, cut_all=False)))

    datasets = pd.DataFrame(corpus_to_tfidf_vector(datasets))
    labels = pd.DataFrame(labels)

    shuffled_indices = np.random.permutation(len(labels))
    test_set_size = int(len(labels) * TEST_RATIO)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return datasets.iloc[train_indices], datasets.iloc[test_indices], \
           labels.iloc[train_indices], labels.iloc[test_indices]


def text_classify(X_train, X_test, y_train, y_test):
    """
    machine learning classifier
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :return:
    """
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
    X_train, X_test, y_train, y_test = prepare_data()
    text_classify(X_train, X_test, y_train, y_test)
