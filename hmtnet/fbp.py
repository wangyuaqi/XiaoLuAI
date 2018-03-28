import copy
import time
import sys

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, datasets

sys.path.append('../')
from hmtnet.cfg import cfg
from hmtnet import data_loader, file_utils, vgg_m_face_bn_dag


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
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

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


def train_gnet(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=25):
    """
    train GNet
    :param model:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param optimizer:
    :param scheduler:
    :param num_epochs:
    :return:
    """
    for epoch in range(num_epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if torch.cuda.is_available():
                model = model.cuda()
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    print('Save trained model...')

    model_path_dir = './model'
    file_utils.mkdirs_if_not_exist(model_path_dir)
    torch.save(model.state_dict(), os.path.join(model_path_dir, 'gnet.pth'))

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available():
            model.cuda()
            labels = labels.cuda()
            outputs = model.forward(Variable(images.cuda()))
        else:
            outputs = model.forward(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def finetune_vgg_m_model(model_ft, train_loader, test_loader, criterion, optimizer, num_epochs=25):
    num_ftrs = model_ft.fc8.out_channels
    model_ft.fc = nn.Linear(num_ftrs, 2)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    for epoch in range(num_epochs):  # loop over the dataset multiple times

        exp_lr_scheduler.step()
        model_ft.train(True)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            if torch.cuda.is_available():
                model_ft = model_ft.cuda()
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model_ft.forward(inputs)
            outputs = outputs.view(-1, model_ft.num_flat_features(outputs))

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 100 == 99:  # print every 200 mini-batches
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0

    print('Finished Training')
    print('Save trained model...')

    model_path_dir = './model'
    file_utils.mkdirs_if_not_exist(model_path_dir)
    torch.save(model_ft.state_dict(), os.path.join(model_path_dir, 'ft_vgg_m.pth'))

    model_ft.train(False)
    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        if torch.cuda.is_available():
            model_ft.cuda()
            labels = labels.cuda()
            outputs = model_ft.forward(Variable(images.cuda()))
        else:
            outputs = model_ft.forward(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


if __name__ == '__main__':
    # gnet = GNet()

    vgg_m_face = vgg_m_face_bn_dag.load_vgg_m_face_bn_dag()
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                             std=[1, 1, 1])
    ])
    gender_dataset = datasets.ImageFolder(root=cfg['gender_base_dir'],
                                          transform=data_transform)
    train_loader, test_loader = data_loader.split_train_and_test_with_py_datasets(data_set=gender_dataset,
                                                                                  batch_size=cfg['batch_size'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(vgg_m_face.parameters(), lr=0.001, weight_decay=1e-4)

    # print('***************************start training GNet***************************')
    # train_gnet(gnet, train_loader, test_loader, criterion, optimizer, scheduler=None, num_epochs=10)

    print('***************************start fine-tuning VGGMFace***************************')
    finetune_vgg_m_model(vgg_m_face, train_loader, test_loader, criterion, optimizer, 10)
