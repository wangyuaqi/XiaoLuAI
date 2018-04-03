from __future__ import print_function

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy


class HMTLoss(nn.Module):

    def __init__(self, target, weight_g, weight_r, weight_a):
        super(HMTLoss, self).__init__()

        self.weight_g = weight_g
        self.weight_r = weight_r
        self.weight_a = weight_a

        self.g_criterion = nn.CrossEntropyLoss()
        self.r_criterion = nn.CrossEntropyLoss()
        self.a_criterion = nn.MSELoss()

    def forward(self, input):
        self.g_loss = self.g_criterion(input, self.target)

        self.output = input

        return self.output

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)

        return self.loss
