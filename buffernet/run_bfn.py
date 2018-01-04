"""
run BufferNet
"""
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim import lr_scheduler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from buffernet.buffered_networks import BufferMLP
from buffernet.utilize import prepare_data, mkdirs_if_not_exist, \
    load_bf_config_by_dataset_name, load_dnn_config_by_dataset_name


def train_bfn(trainloader, net=BufferMLP(), model_path_dir='./model/'):
    """
    train buffered neural networks
    :param dataset_name:
    :param net:
    :param model_path_dir:
    :return:
    """
    # net.apply(init_weights)
    print(net)
    bf_cfg = load_dnn_config_by_dataset_name()

    print('load config : %s ' % str(bf_cfg))
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(net.parameters(), lr=bf_cfg['lr_init'], momentum=bf_cfg['momentum'])
    learning_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    for epoch in range(bf_cfg['epoch']):  # loop over the dataset multiple times
        running_loss = 0.0
        learning_rate_scheduler.step()
        net.train(True)
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            if torch.cuda.is_available():
                net = net.cuda()
                inputs = inputs.cuda()
                labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training\n')
    mkdirs_if_not_exist(model_path_dir)
    torch.save(net.state_dict(), os.path.join(model_path_dir, 'bf-mlp-svhn.pth'))


def test_bfn(testloader, net=BufferMLP(), model_path='./model/bf-mlp-svhn.pth'):
    """
    test buffered neural networks
    :param testloader:
    :param net:
    :param model_path:
    :return:
    """
    net.load_state_dict(torch.load(model_path))
    print('Start Testing')
    net.train(False)

    correct = 0
    total = 0
    for data in testloader:
        data, label = data
        if torch.cuda.is_available():
            net.cuda()
            data = Variable(data).cuda()
            label = Variable(label).cuda()

        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label.data).sum()

    print('Accuracy of the network on this dataset: %f %%' % (
            100 * correct / total))


if __name__ == '__main__':
    bf_cfg = load_bf_config_by_dataset_name("MNIST", "SVHN")
    transform = transforms.Compose(
        [
            transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))])
    trainloader, testloader = prepare_data(bf_cfg)
    train_bfn(trainloader, net=BufferMLP())
    test_bfn(testloader, net=BufferMLP())
