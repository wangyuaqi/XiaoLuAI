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

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_w2v(train=True):
    """
    get word2vec representation
    :return:
    """
    documents = []
    rates = []

    print('loading corpus...')
    # for xlsx in ['./数学之美.xlsx', './数据挖掘导论.xlsx', './数据挖掘概念与技术.xlsx', './机器学习.xlsx']:
    for xlsx in ['./数据挖掘概念与技术.xlsx']:
        df = pd.read_excel(xlsx, index_col=None)
        documents += df['Comment'].tolist()
        rates += df['Rate'].tolist()
        # print(df.loc[:, ['Comment', 'Rate']])

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
        model = Word2Vec(texts, size=300, window=5, min_count=1, workers=4)
        model.save('./doubanbook.model')

    else:
        print('loading pretrained word2vec model...')
        model = Word2Vec.load('./doubanbook.model')

    # print(model.wv['数学'])
    # similarity = model.wv.similarity('算法', '机器学习')
    # print(similarity)
    features = []

    for text in texts:
        features.append(np.array([model.wv[tx].tolist() for tx in text]).mean(axis=0).tolist())

    return np.array(features), np.array(rates)


if __name__ == '__main__':
    X, y = get_w2v(True)
    print(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svc = svm.SVC()
    svc.fit(X_train, y_train)
    score = svc.score(X_test, y_test)
    print('Acc on test set is %d' % score)
