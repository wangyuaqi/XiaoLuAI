# Face Score

## Introduction
This repo is used for machine learning based facial beauty prediction. More details can be found at [my article at Zhihu](https://zhuanlan.zhihu.com/p/29399781).

## Benchmark
We use [SCUT-FBP](http://www.hcii-lab.net/data/scut-fbp/en/introduce.html) and [Female Facial Beauty Dataset (ECCV2010) v1.0](https://www.researchgate.net/publication/261595808_Female_Facial_Beauty_Dataset_ECCV2010_v10) dataset to validate our algorithm.


## Inference
After train a model, you may want to make prediction by using this pre-trained model, while TensorFlow's official tutorial does not contain too much about it.
If you seek for some help, [this tutorial](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) may do you a big favor.


## Result
![performance metric](https://pic2.zhimg.com/v2-78ca5af68b079f8036c8af62d66c4241_r.jpg)
![test pic](https://pic2.zhimg.com/50/v2-6acf96c4eb15df795965573b6c4ff8c5_hd.jpg)

For more details, please read my article at [Zhihu](https://zhuanlan.zhihu.com/p/29399781).

## Features
Your can run [this script](./wechat_face_rank.py) to wear a Christmas hat and rank your face beauty among your WeChat friends.
More details can be found at my [Zhihu Answer](https://www.zhihu.com/question/264485365/answer/282126327).
![wechar_share](./wechat_share.png) 

## Note
The face data can only be used for **research**, no other usage is permitted! 
