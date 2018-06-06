"""
Deep Learning for Sentiment Analysis on DouBan Book
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from gensim.models import Word2Vec
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

W2V_DIMENSION = 300
BATCH_SIZE = 16
EPOCH = 20


def get_w2v(train=True):
    """
    get word2vec representation
    :return:
    """
    documents = []
    rates = []

    print('loading corpus...')
    # for xlsx in ['./数学之美.xlsx', './数据挖掘导论.xlsx', './数据挖掘概念与技术.xlsx', './机器学习.xlsx']:
    for xlsx in ['./谁的青春不迷茫.xlsx']:
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

    print(texts)
    if train:
        print('training word2vec...')
        model = Word2Vec(texts, size=W2V_DIMENSION, window=5, min_count=1, workers=4, iter=20)
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
            labels.append(rate_label[i])

    return np.array(features), np.array(labels)


def svm_senti(X_train, y_train, X_test, y_test):
    """
    train sentiment classifier based on SVM
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    svc = svm.SVC(C=1)
    svc.fit(X_train, y_train)
    cm = confusion_matrix(svc.predict(X_test), y_test)
    print(cm)
    f1 = f1_score(y_test, svc.predict(X_test), average='macro')
    print('F1 score: ', f1)
    acc = accuracy_score(y_test, svc.predict(X_test))
    print('Accuracy: ', acc)


class BiLSTM(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size, dropout=0.5):
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.dropout = dropout
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # first is the hidden h
        # second is the cell c
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            return (torch.zeros(2, self.batch_size, self.hidden_dim).to(device),
                    torch.zeros(2, self.batch_size, self.hidden_dim).to(device))
        else:
            return (torch.zeros(2, self.batch_size, self.hidden_dim),
                    torch.zeros(2, self.batch_size, self.hidden_dim))

    def forward(self, sentence):
        x = self.embeddings(sentence).view(len(sentence), self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)

        return log_probs


class DoubanCommentsDataset(Dataset):
    """
    Douban Comments Dataset
    """

    def __init__(self, X_train, y_train, X_test, y_test, train=True):
        if train:
            self.X = X_train
            self.y = y_train
        else:
            self.X = X_test
            self.y = y_test

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {'ft': self.X[idx], 'senti': self.y[idx]}

        return sample


def get_accuracy(truth, pred):
    """
    calculate accuracy
    :param truth:
    :param pred:
    :return:
    """
    assert len(truth) == len(pred)
    right = 0
    for i in range(len(truth)):
        if truth[i] == pred[i]:
            right += 1.0
    return right / len(truth)


def bilstm_senti(X_train, X_test, y_train, y_test):
    bilstm = BiLSTM(embedding_dim=W2V_DIMENSION, hidden_dim=150, vocab_size=len(X_train),
                    label_size=3, batch_size=BATCH_SIZE)
    optimizer = optim.Adam(bilstm.parameters(), lr=1e-3)
    loss_function = nn.NLLLoss()

    train_dataloader = DataLoader(DoubanCommentsDataset(X_train, X_test, y_train, y_test, train=True),
                                  batch_size=BATCH_SIZE,
                                  shuffle=True, num_workers=4)

    test_dataloader = DataLoader(DoubanCommentsDataset(X_train, X_test, y_train, y_test, train=False),
                                 batch_size=BATCH_SIZE,
                                 shuffle=False, num_workers=4)

    print('Training BiLSTM...')
    for epoch in range(EPOCH):
        bilstm.train()
        avg_loss = 0.0
        truth_res = []
        pred_res = []
        count = 0
        for i, data in enumerate(train_dataloader, 0):
            ft, senti = data['ft'], data['senti']
            senti.data.sub_(1)
            truth_res += list(senti.data)
            bilstm.batch_size = len(senti.data)
            bilstm.hidden = bilstm.init_hidden()
            pred = bilstm(ft)
            pred_label = pred.data.max(1)[1].numpy()
            pred_res += [x for x in pred_label]
            bilstm.zero_grad()
            loss = loss_function(pred, senti)
            avg_loss += loss.item()
            count += 1
            loss.backward()
            optimizer.step()
            if i % 20 == 19:
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, avg_loss / 20))

        avg_loss /= len(X_train)
        acc = get_accuracy(truth_res, pred_res)

    print('Finished Training')
    bilstm.eval()

    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data
            outputs = bilstm(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(3):
        print('Accuracy of %d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    X, y = get_w2v(False)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bilstm_senti(X_train, X_test, y_train, y_test)
