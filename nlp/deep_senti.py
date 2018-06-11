"""
Deep Learning for Sentiment Analysis on DouBan Comments
"""

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TFIDF_FEATURE_NUM = 200
W2V_DIMENSION = 300
D2V_DIMENSION = 100
BATCH_SIZE = 16
EPOCH = 20

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {'ft': self.X[idx], 'senti': self.y[idx]}

        return sample


class RNN(nn.Module):
    def __init__(self, input_size=30, hidden_size=128, num_layers=2, num_classes=3):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out


def rnn_senti(X_train, y_train, X_test, y_test):
    rnn = RNN().to(device)
    criterion = nn.CrossEntropyLoss()
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
            loss = criterion(outputs, senti)

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
