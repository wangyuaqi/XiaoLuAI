"""
a PyTorch porn image recognition implementation powered by deep convolutional neural networks
"""
import os

import torch
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def prepare_data(root_dir='/media/lucasx/Document/DataSet/CV/TrainAndTestPornImages', type='train'):
    """
    build dataloader
    :param root_dir:
    :return:
    """
    data_transform = transforms.Compose([
        transforms.RandomSizedCrop(32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    hymenoptera_dataset = datasets.ImageFolder(root=os.path.join(root_dir, type),
                                               transform=data_transform)
    dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,
                                                 batch_size=4, shuffle=True,
                                                 num_workers=4)

    return dataset_loader


class PRNet(nn.Module):
    """Porn Recognition Network"""

    def __init__(self):
        super(PRNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
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


def train_and_test(trainloader, testloader, model_path_dir='../model/'):
    """
    train PRNet
    :param dataloader:
    :return:
    """
    net = PRNet()
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-4, momentum=0.9)

    print('Start training CNN...')
    for epoch in range(200):  # loop over the dataset multiple times

        running_loss = 0.0
        for i_batch, sample_batched in enumerate(trainloader):
            inputs, labels = sample_batched
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            optimizer.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i_batch % 100 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 2000))
                running_loss = 0.0

                print("Save model to %s/prnet_vot-otb.pth" % model_path_dir)

                if not os.path.isdir(model_path_dir) or not os.path.exists(model_path_dir):
                    os.makedirs(model_path_dir)
                torch.save(net.state_dict(), os.path.join(model_path_dir, 'prnet.pth'))

    print('Finish Training CNN...')

    correct = 0
    total = 0
    for data in testloader:
        images, labels = data
        if torch.cuda.is_available():
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        else:
            images = Variable(images)
            labels = Variable(labels)

        outputs = net(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))


if __name__ == '__main__':
    trainloader = prepare_data(type='train')
    testloader = prepare_data(type='test')
    train_and_test(trainloader, testloader)