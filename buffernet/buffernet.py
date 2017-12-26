import os
import sys

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from buffernet.models import *
from buffernet.utilize import mkdirs_if_not_exist, load_config


def main(dataset_name="MNIST", net=LeNet(), model_path_dir='./model/'):
    print(net)
    for _ in load_config()['dataset']:
        if _['name'] == dataset_name:
            cfg = _
            break

    print('load config : %s ' % str(cfg))
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [
            # transforms.RandomSizedCrop(32),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if cfg['name'] == "MNIST":
        trainset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)

        testset = torchvision.datasets.MNIST(root=cfg['root'], download=False, train=False, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg['batch_size'], shuffle=False, num_workers=2)
    elif cfg['name'] == "SVHN":
        trainset = torchvision.datasets.SVHN(root=cfg['root'], split="train", download=False, transform=transforms)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)

        testset = torchvision.datasets.SVHN(root=cfg['root'], split="test", download=False, transform=transforms)
        testloader = torch.utils.data.DataLoader(testset, batch_size=cfg['batch_size'], shuffle=True, num_workers=2)
    else:
        print('Invalid dataset !!')

    optimizer = optim.SGD(net.parameters(), lr=cfg['lr'], momentum=cfg['momentum'])

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
                net = net.cuda()

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
    print('Start Testing')

    correct = 0
    total = 0
    for data in testloader:
        data, label = data
        if torch.cuda.is_available():
            data = Variable(data).cuda()
            label = Variable(label).cuda()

        output = net(data)
        _, predicted = torch.max(output.data, 1)
        total += label.size(0)
        correct += (predicted == label.data).sum()

    print('Accuracy of the network on the MNIST dataset: %f %%' % (
            100 * correct / total))

    mkdirs_if_not_exist(model_path_dir)
    torch.save(net.state_dict(), os.path.join(model_path_dir, 'prnet.pth'))


if __name__ == '__main__':
    main("MNIST", net=MLP())
