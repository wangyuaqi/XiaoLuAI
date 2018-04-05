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

    def __init__(self, weight_g=1, weight_r=1, weight_a=1):
        super(HMTLoss, self).__init__()

        self.weight_g = weight_g
        self.weight_r = weight_r
        self.weight_a = weight_a

        self.g_criterion = nn.CrossEntropyLoss()
        self.r_criterion = nn.CrossEntropyLoss()
        self.a_criterion = nn.MSELoss()

    def forward(self, g_pred, g_gt, r_pred, r_gt, a_pred, a_gt):
        g_loss = self.g_criterion(g_pred, g_gt)
        r_loss = self.r_criterion(r_pred, r_gt)
        a_loss = self.a_criterion(a_pred, a_gt)

        hmt_loss = self.weight_g * g_loss + self.weight_r * r_loss + self.weight_a * a_loss

        return hmt_loss

    # def backward(self, retain_graph=True):
    #     self.loss.backward(retain_graph=retain_graph)
    #
    #     return self.loss
