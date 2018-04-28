import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms

from bicnn.data_loder import ScutFBPDataset

sys.path.append('../')
from bicnn.utils import mkdirs_if_not_exist
from bicnn.cfg import cfg


def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, scheduler, num_epochs=25,
                inference=False):
    print('Start training Bi-CNN...')
    model = model.float()
    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            model.train()
            scheduler.step()

            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                inputs, labels = data['image'], data['score']

                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    model = model.to(device)
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                if torch.cuda.device_count() > 1:
                    print("Let's use", torch.cuda.device_count(), "GPUs!")
                    model = nn.DataParallel(model)

                # zero the parameter gradients
                optimizer.zero_grad()

                inputs = inputs.float()
                labels = labels.float().view(cfg['batch_size'], 1)

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

        print('Finished training Bi-CNN...\n')
        print('Saving trained model...')
        model_path_dir = './model'
        mkdirs_if_not_exist(model_path_dir)
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'bi-cnn.pth'))
        print('Deep beauty model has been saved successfully~')

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/bi-cnn.pth')))

    model.eval()

    predicted_labels = []
    gt_labels = []
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

        predicted_labels += outputs.cpu().data.numpy().tolist()
        gt_labels += labels.cpu().numpy().tolist()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of Bi-CNN is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of Bi-CNN is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of Bi-CNN is {0}===================='.format(pc))


def ft_deep_beauty_model():
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model_ft = model_ft.to(device)

    criterion = nn.MSELoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=20, gamma=0.1)

    df = pd.read_excel('./cvsplit/SCUT-FBP.xlsx', sheet_name='Sheet1')
    X_train, X_test, y_train, y_test = train_test_split(df['Image'].tolist(), df['Attractiveness label'],
                                                        test_size=0.2, random_state=0)

    train_dataset = ScutFBPDataset(f_list=X_train, f_labels=y_train, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    test_dataset = ScutFBPDataset(f_list=X_test, f_labels=y_test, transform=transforms.Compose([
        transforms.ColorJitter(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomRotation(30),
        transforms.ToTensor()
    ]))

    train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'],
                                 shuffle=False, num_workers=4)

    train_model(model=model_ft, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                criterion=criterion, optimizer=optimizer_ft, scheduler=exp_lr_scheduler, num_epochs=30, inference=False)


if __name__ == '__main__':
    ft_deep_beauty_model()
