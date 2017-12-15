"""
a branched deep learning regression algorithm for facial beauty prediction
"""
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class EccvFbNet(nn.Module):
    def __init__(self):
        super(EccvFbNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(200704, 102400)
        self.fc2 = nn.Linear(102400, 1)

    def forward(self, x):
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def main(conv_features, label):
    net = EccvFbNet()
    output = net(conv_features)
    target = Variable(torch.arange(1, 11))  # a dummy target, for example
    criterion = nn.MSELoss()

    loss = criterion(output, target)


if __name__ == '__main__':
    pass
