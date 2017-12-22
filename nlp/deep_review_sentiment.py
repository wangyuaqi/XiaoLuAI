"""
Intro: review sentiment classification based on deep learning
Note: unfinished!
"""
import logging
import os

from lxml import etree

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

BASE_CORPUS_DIR = 'E:/DataSet/NLP/SentimentAnalysis/domain_sentiment_data/sorted_data_acl'
USER_DICT = './user_dict.txt'
STOP_WORDS = './stopwords.txt'


def read_document(document_filepath):
    """
    read document content from text file and return in one line
    :param document_filepath:
    :return:
    """
    with open(document_filepath, mode='rt', encoding='UTF-8') as f:
        return ''.join(f.readlines())


def parse_xml(review_xml_path):
    tree = etree.parse(review_xml_path)
    rating = tree.xpath('//rating/text()')
    print(rating)


if __name__ == '__main__':
    for doc_file in os.listdir(os.path.join(BASE_CORPUS_DIR, 'books')):
        parse_xml(os.path.join(BASE_CORPUS_DIR, 'books', 'negative.review'))
