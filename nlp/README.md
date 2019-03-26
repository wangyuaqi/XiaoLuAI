# Natural Language Process Module

## Introduction
This module is designed for NLP task, it includes several NLP tasks:
* [Text Classification](./text_classifier.py)
* [Deep Learning for Sentiment Analysis](./deep_senti.py)

## Corpora
In order to make use of NLP module, you'd better download the corpora data firstly. The following corpora data is recommended:
* [Fudan NLP Text Corpora](https://share.weiyun.com/67ac1ff60864c564a86181e5e84cd2e4)  
* [Douban Comments](./douban_comments.7z)

## Performance
* This unoptimized code reaches accuracy **82.09184%** by [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree) with the feature vector dimension of **2000** on 
[Fudan NLP Text Corpora Benchmark](https://share.weiyun.com/67ac1ff60864c564a86181e5e84cd2e4) which contains **20** classes. 
* The deep learning-based algorithm achieves a F1 score with 0.7808 and Accuracy with 0.7937 on [Douban Comments](./douban_comments.7z). 

We believe it'll be much improved 
when precisely picked [user dict](./user_dict.txt) and [stopwords](./stopwords.txt) are given.


## Note
For more details, please contact [Yu Cheng](https://github.com/fypns) for more details.
