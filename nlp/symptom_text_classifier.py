import os

import jieba
import jieba.analyse
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors.nearest_centroid import NearestCentroid

USER_DICT = './user_dict.txt'
STOP_WORDS = './stopwords.txt'
FEATURE_NUM = 2000
W2V_DIMENSION = 300
TEST_RATIO = 0.2
STOP_DENOTATION = [u'《', u'》', u'%', u'\n', u'、', u'=', u' ', u'+', u'-', u'~', u'', u'#', u'＜', u'＞']


def get_disease_map():
    df = pd.read_excel("./tcm_data.xlsx")
    diseases = df['诊断']

    idx = 0

    mp = {}

    for i, disease in enumerate(diseases):
        if disease.strip() not in mp.keys():
            mp[disease.strip()] = idx
            idx += 1

    for k, v in mp.items():
        print("{0},{1}".format(k, v))


def read_corpus():
    """
    read corpus from Excel file and cut words
    :return:
    """
    df = pd.read_excel("./tcm_data.xlsx")

    symptom_texts = df['现病史']
    diseases = df['诊断']

    documents, labels = [], []

    disease_map = {}
    d = pd.read_csv('./disease_map.csv')
    for i in range(len(d)):
        disease_map[d['name'][i]] = d['id'][i]

    for i, symptom_text in enumerate(symptom_texts):
        documents.append(' '.join(jieba.cut(symptom_text, cut_all=False)))
        labels.append(disease_map[diseases[i]])

    print('loading corpus...')

    texts = []

    jieba.load_userdict('./user_dict.txt')
    jieba.analyse.set_stop_words('./stopwords.txt')
    stopwords = [_.replace('\n', '') for _ in open('./stopwords.txt', encoding='utf-8').readlines()]

    for doc in documents:
        words_in_doc = list(jieba.cut(doc.strip()))
        words_in_doc = list(filter(lambda w: w not in STOP_DENOTATION + stopwords, words_in_doc))
        texts.append(words_in_doc)

    return texts, labels


def get_w2v(texts, label, train=True):
    """
    get word2vec representation
    :return:
    """
    print(texts)
    if train:
        print('training word2vec...')
        model = Word2Vec(texts, size=W2V_DIMENSION, window=5, min_count=1, workers=4, iter=20)
        if not os.path.isdir('./model') or not os.path.exists('./model'):
            os.makedirs('./model')
        model.save('./model/disease_w2v.model')

    else:
        print('loading pretrained word2vec model...')
        model = Word2Vec.load('./model/disease_w2v.model')

    # print(model.wv['数学'])
    # similarity = model.wv.similarity('算法', '机器学习')
    # print(similarity)
    features = list()
    labels = list()

    for i in range(len(texts)):
        f = np.array([model.wv[tx] for tx in texts[i]]).mean(axis=0).flatten().tolist()
        if len(f) == W2V_DIMENSION:
            features.append(f)
            labels.append(label[i])

    return np.array(features), np.array(labels)


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

    df = pd.read_excel("./tcm_data.xlsx")

    symptom_texts = df['现病史']
    diseases = df['诊断']

    disease_map = {}
    d = pd.read_csv('./disease_map.csv')
    for i in range(len(d)):
        disease_map[d['name'][i]] = d['id'][i]

    for i, symptom_text in enumerate(symptom_texts):
        datasets.append(' '.join(jieba.cut(symptom_text, cut_all=False)))
        labels.append(disease_map[diseases[i]])

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
    # print('=' * 100)
    # print('start launching MLP Classifier......')
    # mlp = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(50, 30, 20, 20, 20, 30, 50), random_state=1)
    # mlp.fit(X_train, y_train)
    # print('finish launching MLP Classifier, the test accuracy is {:.5%}'.format(mlp.score(X_test, y_test)))

    print('=' * 100)
    print('start launching SVM Classifier......')
    svc = svm.SVC(decision_function_shape='ovo')
    svc.fit(X_train, y_train)
    print('finish launching SVM Classifier, the test accuracy is {:.5%}'.format(
        accuracy_score(svc.predict(X_test), y_test)))

    print('=' * 100)
    print('start launching Decision Tree Classifier......')
    dtree = tree.DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    print('finish launching Decision Tree Classifier, the test accuracy is {:.5%}'.format(
        accuracy_score(dtree.fit(X_test), y_test)))

    print('=' * 100)
    print('start launching KNN Classifier......')
    knn = NearestCentroid()
    knn.fit(X_train, y_train)
    print('finish launching KNN Classifier, the test accuracy is {:.5%}'.format(
        accuracy_score(knn.predict(X_test), y_test)))

    print('=' * 100)
    print('start launching Random Forest Classifier......')
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    print('finish launching Random Forest Classifier, the test accuracy is {:.5%}'.format(
        accuracy_score(rf.fit(X_test), y_test)))


if __name__ == '__main__':
    # X_train, X_test, y_train, y_test = prepare_data()

    texts, labels = read_corpus()
    X, y = get_w2v(texts, labels, train=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_RATIO)
    text_classify(X_train, X_test, y_train, y_test)

    # get_disease_map()
