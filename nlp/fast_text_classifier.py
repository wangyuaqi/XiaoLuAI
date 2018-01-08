import os

import numpy as np
import jieba
import fasttext

BASE_DIR = 'E:/DataSet/NLP/tc-corpus-answer/answer'


def preprocessing(base_dir, train_ratio=0.3):
    # train = open('./train.txt', mode='w')
    # test = open('./test.txt', mode='w')
    train_list = []
    test_list = []
    labels = [_ for _ in os.listdir(BASE_DIR)]
    for label in labels:
        tmp_list = []
        for txt in os.listdir(os.path.join(BASE_DIR, label)):
            with open(os.path.join(BASE_DIR, label, txt), mode='rt', encoding='latin1') as f:
                text = ''.join(f.readlines())

            seg_text = jieba.cut(text.replace('\t', '').replace('\n', ' '))
            line = ' '.join(seg_text).encode('utf-8').decode('utf-8') + '\t__label__' + label + '\n'
            tmp_list.append(line)

            shuffled_indices = np.random.permutation(len(tmp_list)).tolist()

        for l in shuffled_indices[0:int(len(shuffled_indices) * train_ratio)]:
            train_list.append(tmp_list[l])
        for _ in tmp_list:
            if _ not in train_list:
                test_list.append(_)

    with open('./train.txt', mode='w', encoding='utf-8') as ftrain, \
            open('./test.txt', mode='w', encoding='utf-8') as ftest:
        ftrain.write(''.join(train_list))
        ftest.write(''.join(test_list))
        ftrain.flush()
        ftest.flush()
        ftrain.close()
        ftest.close()


def train():
    classifier = fasttext.supervised('train.txt', 'model', label_prefix='__label__')
    result = classifier.test('test.txt')
    print('Precision:', result.precision)
    print('Recall:', result.recall)


def predict(text):
    model = fasttext.load_model('./model.bin')
    classifier = fasttext.load_model('./model.bin', label_prefix='__label__')
    labels = classifier.predict_proba(text)
    print(labels)


if __name__ == '__main__':
    preprocessing(BASE_DIR, 0.7)
    train()
