import torch.nn as nn
import torch.nn.functional as F


class HMTNet(nn.Module):
    """
    definition of HMTNet
    """

    def __init__(self):
        super(HMTNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class RNet(nn.Module):
    """
    definition of RaceNet
    """

    def __init__(self):
        super(RNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        # self.gfc1 = nn.Linear(56 * 56 * 256, 4096)
        # self.gfc2 = nn.Linear(4096, 2)

        self.gfc1 = nn.Linear(56 * 56 * 2, 32)
        self.gfc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool4(self.conv4(self.conv3(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))

        x = self.gfc2(F.relu(self.gfc1(x)))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class GNet(nn.Module):
    """
    definition of GenderNet
    """

    def __init__(self):
        super(GNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(128, 2, kernel_size=3, stride=1, padding=1)

        # self.gfc1 = nn.Linear(56 * 56 * 256, 4096)
        # self.gfc2 = nn.Linear(4096, 2)

        self.gfc1 = nn.Linear(56 * 56 * 2, 32)
        self.gfc2 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool4(self.conv4(self.conv3(x)))
        x = F.relu(self.conv5(x))
        x = x.view(-1, self.num_flat_features(x))

        x = self.gfc2(F.relu(self.gfc1(x)))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features
