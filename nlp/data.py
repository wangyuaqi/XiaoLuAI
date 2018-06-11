"""
text pre-processing and loading
"""
import sys

import numpy as np
import pandas as pd

from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from torch.utils.data import Dataset
from torchtext import data

sys.path.append('../')
from nlp.config import TFIDF_FEATURE_NUM, W2V_DIMENSION, D2V_DIMENSION


def read_corpus():
    """
    read corpus from Excel file and cut words
    :return:
    """
    documents = []
    rates = []

    print('loading corpus...')
    # for xlsx in ['./谁的青春不迷茫.xlsx']:
    for xlsx in ['Python核心编程第二版.xlsx', './谁的青春不迷茫.xlsx', 'HeadFirst数据分析.xlsx', '大数据时代.xlsx',
                 '你若安好便是晴天.xlsx', '悲伤逆流成河.xlsx']:
        df = pd.read_excel(xlsx, index_col=None)
        df = df.dropna(how='any')
        documents += df['Comment'].tolist()
        rates += df['Rate'].tolist()

    rate_label = []
    for _ in rates:
        if 1 <= _ <= 2:
            rate_label.append(0)
        elif _ == 3:
            rate_label.append(1)
        else:
            rate_label.append(2)

    print('tokenizer starts working...')

    texts = []
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

    return texts, rate_label


def load_torchtext():
    texts = data.Field(sequential=True, lower=False, batch_first=True, fix_length=30)
    labels = data.LabelField(sequential=False)


def corpus_to_tfidf_vector(texts, rate_label):
    """
    convert segmented corpus in a list into TF-IDF array
    :param texts:
    :return:
    """
    vectorizer = CountVectorizer(min_df=1, max_features=TFIDF_FEATURE_NUM)
    transformer = TfidfTransformer(smooth_idf=True)
    corpus_list = [''.join(text) for text in texts]
    X = vectorizer.fit_transform(corpus_list)
    tfidf = transformer.fit_transform(X)

    return tfidf.toarray(), rate_label


def get_w2v(texts, rate_label, train=True):
    """
    get word2vec representation
    :return:
    """
    print(texts)
    if train:
        print('training word2vec...')
        model = Word2Vec(texts, size=W2V_DIMENSION, window=5, min_count=1, workers=4, iter=20)
        model.save('./doubanbook_w2v.model')

    else:
        print('loading pretrained word2vec model...')
        model = Word2Vec.load('./doubanbook_w2v.model')

    # print(model.wv['数学'])
    # similarity = model.wv.similarity('算法', '机器学习')
    # print(similarity)
    features = list()
    labels = list()

    for i in range(len(texts)):
        f = np.array([model.wv[tx] for tx in texts[i]]).mean(axis=0).flatten().tolist()
        if len(f) == W2V_DIMENSION:
            features.append(f)
            labels.append(rate_label[i])

    return np.array(features), np.array(labels)


def get_d2v(words_list, labels, train=True):
    """
    get doc2vec representation
    :param words_list:
    :param labels:
    :param train:
    :return:
    """
    if train:
        print('training doc2vec...')
        documents = []
        for i in range(len(words_list)):
            documents.append(TaggedDocument(words_list[i], [labels[i]]))

        model = Doc2Vec(size=D2V_DIMENSION, min_count=1, workers=4)
        model.build_vocab(documents)
        model.train(documents, total_examples=model.corpus_count, epochs=20)
        model.save('./doubanbook_d2v.model')
    else:
        print('loading pretrained doc2vec model...')
        model = Doc2Vec.load('./doubanbook_d2v.model')

    features = list()

    for words in words_list:
        f = model.infer_vector(words)
        if len(f) == D2V_DIMENSION:
            features.append(f)

    return np.array(features), np.array(labels)


class DoubanCommentsDataset(Dataset):
    """
    Douban Comments Dataset
    """

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {'ft': self.X[idx], 'senti': self.y[idx]}

        return sample
