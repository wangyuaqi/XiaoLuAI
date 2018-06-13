"""
Deep Learning for Sentiment Analysis on DouBan Comments
"""
import logging
import sys
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.functional as F
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from torch.utils.data import Dataset

sys.path.append('../')
from nlp.data import read_corpus, get_w2v, DoubanCommentsDataset
from nlp.models import DoubanRNN, AutoEncoder
from nlp.config import *

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def unsupervised_pretrain(data_loader):
    """
    train Deep AutoEncoder
    :param data_loader: 
    :return: 
    """
    autoencoder = AutoEncoder().to(device)
    autoencoder.float()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-5, weight_decay=1e-7)
    mse = nn.MSELoss()

    print('start training Deep AutoEncoder...')
    for epoch in range(8):
        running_loss = 0.0
        for i, data in enumerate(data_loader):
            ft = data['ft'].to(device)
            ft = ft.float()

            optimizer.zero_grad()
            encoded, decoded = autoencoder(ft)

            loss = mse(decoded, ft)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 50 == 49:  # print every 50 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0

    model_path = './model/autoencoder.pth'
    if not os.path.isdir('./model') or not os.path.exists('./model'):
        os.makedirs('./model')
    torch.save(autoencoder.state_dict(), model_path)

    print('Finished Training')


def deep_ft_extract(X):
    """
    extract deep features from a pre-trained DeepAE
    :param X:
    :return:
    """
    print('Loading pre-trained Deep AutoEncoder...')
    autoencoder = AutoEncoder()
    autoencoder.load_state_dict(torch.load('./model/autoencoder.pth'))
    X_new = []

    for x in X:
        for idx, module in autoencoder.named_children():
            if idx != 'decoder':
                x = torch.from_numpy(np.array(x)).unsqueeze(0).float()
                x = module(x)
                # print(x.numpy().tolist())
                X_new.append(x.detach().numpy().flatten().tolist())

    return X_new


def svm_senti(X_train, y_train, X_test, y_test):
    """
    train sentiment classifier based on SVM
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    svc = svm.SVC(C=1, kernel='rbf')
    svc.fit(X_train, y_train)
    cm = confusion_matrix(svc.predict(X_test), y_test)
    print(cm)
    f1 = f1_score(y_test, svc.predict(X_test), average='macro')
    print('F1 score: ', f1)
    acc = accuracy_score(y_test, svc.predict(X_test))
    print('Accuracy: ', acc)

    joblib.dump(svc, './model/svc.pkl')


def rand_forest_senti(X_train, y_train, X_test, y_test):
    """
    train sentiment classifier based on Random Forest
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """
    rf = RandomForestClassifier(n_estimators=20)
    rf.fit(X_train, y_train)
    cm = confusion_matrix(rf.predict(X_test), y_test)
    print(cm)
    f1 = f1_score(y_test, rf.predict(X_test), average='macro')
    print('F1 score: ', f1)
    acc = accuracy_score(y_test, rf.predict(X_test))
    print('Accuracy: ', acc)

    joblib.dump(rf, './model/rand_forest.pkl')


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
    X, y = get_w2v(texts, rate_label, True)
    # X, y = get_d2v(texts, rate_label, True)
    print(pd.Series(y).value_counts())

    data_loader = torch.utils.data.DataLoader(DoubanCommentsDataset(X, y), batch_size=BATCH_SIZE, shuffle=True,
                                              num_workers=4)
    unsupervised_pretrain(data_loader)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_train_ae = deep_ft_extract(X_train)
    X_test_ae = deep_ft_extract(X_test)

    pca = PCA(n_components=30)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    X_train = np.concatenate((X_train, X_train_ae), axis=1)
    X_test = np.concatenate((X_test, X_test_ae), axis=1)

    svm_senti(X_train, y_train, X_test, y_test)
    # rand_forest_senti(X_train, y_train, X_test, y_test)
    # rnn_senti(X_train, y_train, X_test, y_test)
