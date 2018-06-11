"""
Deep Learning for Sentiment Analysis on DouBan Comments
"""
import logging
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

sys.path.append('../')
from nlp.data import read_corpus, get_w2v, DoubanCommentsDataset
from nlp.models import DoubanRNN
from nlp.config import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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


def rnn_senti(X_train, y_train, X_test, y_test):
    rnn = DoubanRNN().to(device)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-4)

    print('prepare datasets for RNN...')

    train_loader = torch.utils.data.DataLoader(DoubanCommentsDataset(X_train, y_train), batch_size=BATCH_SIZE,
                                               shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(DoubanCommentsDataset(X_test, y_test), batch_size=BATCH_SIZE,
                                              shuffle=False, num_workers=4)

    # Train the model
    total_step = len(train_loader)
    rnn.train()
    for epoch in range(EPOCH):
        for i, data in enumerate(train_loader):
            ft, senti = data['ft'].reshape(-1, BATCH_SIZE, 30).to(device), data['senti'].to(device)

            # Forward pass
            outputs = rnn.forward(ft)
            loss = F.nll_loss(outputs, senti)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch + 1, EPOCH, i + 1, total_step, loss.item()))

    # Test the model
    rnn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for data in test_loader:
            ft, senti = data['ft'].to(device), data['senti'].to(device)
            outputs = rnn(ft)
            _, predicted = torch.max(outputs.data, 1)
            total += senti.size(0)
            correct += (predicted == senti).sum().item()

        print('Test Accuracy of the model on test set: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    texts, rate_label = read_corpus()
    # X, y = corpus_to_tfidf_vector(texts, rate_label)
    X, y = get_w2v(texts, rate_label, False)

    print(pd.Series(y).value_counts())

    # X, y = get_d2v(texts, rate_label, True)
    print(X.shape)
    pca = PCA(n_components=30)
    X = pca.fit_transform(X)
    print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # svm_senti(X_train, y_train, X_test, y_test)
    rnn_senti(X_train, y_train, X_test, y_test)
