import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

sys.path.append('../')
from scene.cfg import cfg
from scene.data_loader import IndoorDataset


def mkdirs_if_not_exist(dir_name):
    """
    create new folder if not exist
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name) or not os.path.exists(dir_name):
        os.makedirs(dir_name)


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                inference=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    if not inference:
        model.train()
        print('Start training ResNet18...')
        for epoch in range(num_epochs):
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data['image'], data['class']

                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                inputs = inputs.float()
                # labels = labels.float().view(cfg['batch_size'], 5)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 50 == 49:  # print every 50 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 50))
                    running_loss = 0.0

        print('Finished training ResNet18...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'resnet18.pth'))
        print('ResNet18 has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/resnet18.pth')))

    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():

        for data in test_dataloader:
            inputs, labels = data['image'], data['class']
            inputs = inputs.to(device)

            outputs = model(inputs)
            outputs = outputs.to("cpu")
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


def run_resnet18_on_indoor():
    """
    train ResNet18 with 67-Class on Indoor dataset
    :return:
    """
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 67)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    train_dataset = IndoorDataset(train=True, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    test_dataset = IndoorDataset(train=False, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4, drop_last=True)

    train_model(model=model_ft, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=50, inference=False)


if __name__ == '__main__':
    run_resnet18_on_indoor()
