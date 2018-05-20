"""
a PyTorch porn image recognition implementation powered by deep convolutional neural networks
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets

CLASS_NUM = 3
EPOCH = 100
BATCH = 16
# IMAGE_SIZE = 224
IMAGE_SIZE = 128
LR_INIT = 1e-3
WEIGHT_DECAY = 1e-5


def prepare_data(root_dir='E:/DataSet/CV/TrainAndTestPornImages', type='train'):
    """
    build dataloader
    :param type: train for trainset, test for testset
    :param root_dir:
    :return:
    """
    data_transform = transforms.Compose([
        transforms.Resize(160),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])
    hymenoptera_dataset = datasets.ImageFolder(root=os.path.join(root_dir, type),
                                               transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                                 batch_size=BATCH, shuffle=True if type == 'train' else False,
                                                 num_workers=4)

    return dataset_loader


class PRNet(nn.Module):
    """Porn Recognition Network"""

    def __init__(self, num_class=3):
        super(PRNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, stride=1)
        self.conv2 = nn.Conv2d(64, 256, 3)
        self.conv3 = nn.Conv2d(256, 512, 3)
        self.conv4 = nn.Conv2d(512, 64, 1)
        self.fc1 = nn.Linear(64 * 8 * 8, num_class, nn.Dropout(0.5))
        # self.fc2 = nn.Linear(256, 3, nn.Dropout(0.5))

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 3)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = self.conv4(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, CLASS_NUM)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(x)

        return F.log_softmax(x)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


def train_and_test(trainloader, testloader, model_path_dir='./model/'):
    """
    train PRNet
    :param model_path_dir:
    :param testloader:
    :param dataloader:
    :return:
    """
    net = PRNet()
    net.train()
    # net = MobileNet()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=LR_INIT, momentum=0.9, weight_decay=WEIGHT_DECAY)

    print('Start training CNN...')
    for epoch in range(EPOCH):  # loop over the dataset multiple times

        running_loss = 0.0
        for i_batch, sample_batched in enumerate(trainloader, 0):
            inputs, labels = sample_batched
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i_batch % 50 == 49:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 50))
                running_loss = 0.0

                print("Save model to %sprnet.pth" % model_path_dir)

                if not os.path.isdir(model_path_dir) or not os.path.exists(model_path_dir):
                    os.makedirs(model_path_dir)
                torch.save(net.state_dict(), os.path.join(model_path_dir, 'prnet.pth'))

    print('Finish Training CNN...')

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    classes = [0, 1, 2]
    class_correct = list(0. for i in range(3))
    class_total = list(0. for i in range(3))
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        c = (predicted == labels).squeeze()
        for i in range(3):
            label = labels[i]
            class_correct[label] += c[i]
            class_total[label] += 1

    print('=' * 100)
    for i in range(3):
        print('Accuracy of type %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))
    print('=' * 100)


def inference(testloader):
    """
    inference
    :param testloader:
    :return:
    """
    net = PRNet()
    # net = MobileNet()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    net.load_state_dict(torch.load('./model/prnet.pth'))
    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        net = net.to(device)
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(3):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(3):
        print('Accuracy of %d : %2d %%' % (i, 100 * class_correct[i] / class_total[i]))


if __name__ == '__main__':
    trainloader = prepare_data(type='train')
    testloader = prepare_data(type='test')
    train_and_test(trainloader, testloader)
    # inference(testloader)
