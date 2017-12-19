import os
import sys
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))


class VggANet(nn.Module):
    """
    VGG A Modification Net
    """

    def __init__(self):
        super(VggANet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.conv4 = nn.Conv2d(256, 256, 3)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 512, 3)
        self.conv7 = nn.Conv2d(512, 512, 3)
        self.conv8 = nn.Conv2d(512, 512, 3)

        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(F.relu(self.conv6(x)), 2)
        x = F.relu(self.conv7(x))
        x = F.max_pool2d(F.relu(self.conv8(x)), 2)

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class LeNet(nn.Module):
    """
    LeNet
    """

    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
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


def load_config(config_json_path='./bfnet_config.json'):
    """
    load configuration in json file
    :param config_json_path:
    :return:
    """
    with open(config_json_path, mode='rt', encoding='UTF-8') as f:
        config = json.load(f)

    return config


def main(dataset_name="SVHN"):
    for _ in load_config()['dataset']:
        if _['name'] == dataset_name:
            cfg = _
            break

    print('load config %s ...' % str(cfg))
    leNet = LeNet()
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.RandomSizedCrop(32),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)

    optimizer = optim.SGD(leNet.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

    for epoch in range(cfg['epoch']):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
                labels = labels.cuda()
                leNet = leNet.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = leNet(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training\n')
    print('Start Testing\n')

    correct = 0
    total = 0
    for data in testloader:
        data, labels = data
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            labels = Variable(labels).cuda()

        outputs = leNet(data)
        _, predicted = torch.max(outputs.cpu().data, 1)
        total += labels.cpu().size(0)
        correct += (predicted == labels.cpu().data).sum()

    print('Accuracy of the network on the 10000 test images: %f %%' % (
        100 * correct / total))


if __name__ == '__main__':
    main("MNIST")
