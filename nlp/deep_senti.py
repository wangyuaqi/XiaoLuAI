"""
Deep Learning for Sentiment Analysis on DouBan Book
"""

import logging

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
W2V_DIMENSION = 300


def get_w2v(train=True):
    """
    get word2vec representation
    :return:
    """
    documents = []
    rates = []

    print('loading corpus...')
    # for xlsx in ['./数学之美.xlsx', './数据挖掘导论.xlsx', './数据挖掘概念与技术.xlsx', './机器学习.xlsx']:
    for xlsx in ['./机器学习.xlsx']:
        df = pd.read_excel(xlsx, index_col=None)
        df = df.dropna(how='any')
        documents += df['Comment'].tolist()
        rates += df['Rate'].tolist()
        # print(df.loc[:, ['Comment', 'Rate']])

    rates = [0 if _ <= 3 else 1 for _ in rates]
    print('tokenizer starts working...')

    texts = []
    import jieba
    import jieba.analyse

    jieba.load_userdict('./user_dict.txt')
    jieba.analyse.set_stop_words('stopwords.txt')
    stopwords = [_.replace('\n', '') for _ in open('./stopwords.txt', encoding='utf-8').readlines()]

    for doc in documents:
        words_in_doc = list(jieba.cut(doc))
        for _ in stopwords:
            if _ in words_in_doc:
                words_in_doc.remove(_)

        texts.append(words_in_doc)

    print(texts)
    if train:
        print('training word2vec...')
        model = Word2Vec(texts, size=W2V_DIMENSION, window=5, min_count=1, workers=4)
        model.save('./doubanbook.model')

    else:
        print('loading pretrained word2vec model...')
        model = Word2Vec.load('./doubanbook.model')

    # print(model.wv['数学'])
    # similarity = model.wv.similarity('算法', '机器学习')
    # print(similarity)
    features = list()
    labels = list()

    for i in range(len(texts)):
        f = np.array([model.wv[tx] for tx in texts[i]]).mean(axis=0).flatten().tolist()
        if len(f) == W2V_DIMENSION:
            features.append(f)
            labels.append(rates[i])

    return np.array(features), np.array(labels)


if __name__ == '__main__':
    X, y = get_w2v(True)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    svc = svm.SVC(C=1)
    svc.fit(X_train, y_train)
    cm = confusion_matrix(svc.predict(X_test), y_test)
    print(cm)
    f1 = f1_score(y_test, svc.predict(X_test), average='macro')
    print('F1 score: ', f1)
    acc = accuracy_score(y_test, svc.predict(X_test))
    print('Accuracy: ', acc)
