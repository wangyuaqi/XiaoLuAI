import sys
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms

sys.path.append('../')
from deepbeauty.utils import mkdirs_if_not_exist
from deepbeauty.cfg import cfg


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                inference=False):
    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data
                if torch.cuda.device_count() > 1:
                    if torch.cuda.is_available():
                        device = torch.device('cuda')
                        model = model.to(device)
                        inputs = inputs.to(device)
                        labels = labels.to(device)

                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
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

        print('Finished Training\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'deep_beauty.pth'))
        print('Deep beauty model has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/deep_beauty.pth')))

    model.train(False)
    correct = 0
    total = 0
    for data in test_dataloader:
        # images, labels = data
        images, labels = data['image'], data['score']
        if torch.cuda.is_available():
            device = torch.device('cuda')
            labels = labels.to(device)
            images = images.to(device)
            outputs = model.forward(images)
        else:
            outputs = model.forward(images)

        outputs = outputs.view(cfg['batch_size'], 1)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)


def ft_deep_beauty_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)
