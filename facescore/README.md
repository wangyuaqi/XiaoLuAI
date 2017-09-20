# Face Score

## Introduction
This repo is used for face score recognition based on machine learning regression algorithm, the training and test dataset are collected throught web crawler in Java of [CVLH](https://github.com/EclipseXuLu/CVLH.git).
 
## Format
We design this by referring to [CIFAR.](http://www.cs.toronto.edu/~kriz/cifar.html)
The binary file generator is [image_util.py](../util/image_util.py)

## Inference
After train a model, you may want to make prediction by using this pre-trained model, while TensorFlow's official tutorial does not contain too much about it.
If you seek for some help, [this tutorial](http://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/) may do you a big favor.

## Note
The face data set can only be used for **research**, no other usage is permitted! 
