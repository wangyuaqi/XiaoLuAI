import copy
import time
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms, datasets

# os.environ['CUDA_VISIBLE_DEVICES'] = 'gpu02'

sys.path.append('../')
from hmtnet.losses import HMTLoss
from hmtnet.models import RNet, GNet, HMTNet
from hmtnet.data_loader import FaceGenderDataset, FaceRaceDataset, FaceDataset
from hmtnet.cfg import cfg
from hmtnet import data_loader, file_utils, vgg_m_face_bn_dag


def train_gnet(model, train_loader, test_loader, criterion, optimizer, num_epochs=25, inference=False):
    """
    train GNet
    :param model:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if not inference:
        model.train(True)
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['label']

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

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/gnet.pth')))

    model.train(False)
    correct = 0
    total = 0
    for data in test_loader:
        # images, labels = data
        images, labels = data['image'], data['label']
        if torch.cuda.is_available():
            model.cuda()
            labels = labels.cuda()
            outputs = model.forward(Variable(images.cuda()))
        else:
            outputs = model.forward(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the network on test images: %f' % (correct / total))


def train_rnet(model, train_loader, test_loader, criterion, optimizer, num_epochs=25, inference=False):
    """
    train GNet
    :param model:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param optimizer:
    :param num_epochs:
    :return:
    """

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)

    if not inference:
        model.train(True)
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['label']

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
        torch.save(model.state_dict(), os.path.join(model_path_dir, 'rnet.pth'))

    else:
        print('Loading pre-trained model...')
        model.load_state_dict(torch.load(os.path.join('./model/rnet.pth')))

    model.train(False)
    correct = 0
    total = 0
    for data in test_loader:
        # images, labels = data
        images, labels = data['image'], data['label']
        if torch.cuda.is_available():
            model.cuda()
            labels = labels.cuda()
            outputs = model.forward(Variable(images.cuda()))
        else:
            outputs = model.forward(Variable(images))

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the network on test images: %f' % (correct / total))


def finetune_vgg_m_model(model_ft, train_loader, test_loader, criterion, num_epochs=25, inference=False):
    """
    fine-tune VGG M Face Model
    :param model_ft:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param num_epochs:
    :param inference:
    :return:
    """
    num_ftrs = model_ft.fc8.in_channels
    model_ft.fc8 = nn.Conv2d(num_ftrs, 2, 1)

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            exp_lr_scheduler.step()
            model_ft.train(True)

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                # inputs, labels = data
                inputs, labels = data['image'], data['label']

                # wrap them in Variable
                if torch.cuda.is_available():
                    model_ft = model_ft.cuda()
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward + backward + optimize
                outputs = model_ft.forward(inputs)
                outputs = (torch.sum(outputs, dim=1) / 2).view(2, 1)
                # outputs = outputs.view(-1, outputs.numel())

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model_ft.state_dict(), os.path.join(model_path_dir, 'ft_vgg_m.pth'))

    else:
        print('Loading pre-trained model...')
        model_ft.load_state_dict(torch.load(os.path.join('./model/ft_vgg_m.pth')))

    model_ft.train(False)
    correct = 0
    total = 0

    # for data in test_loader:
    for i, data in enumerate(test_loader, 0):
        # images, labels = data
        images, labels = data['image'], data['label']
        if torch.cuda.is_available():
            model_ft = model_ft.cuda()
            labels = labels.cuda()
            outputs = model_ft.forward(Variable(images.cuda()))
        else:
            outputs = model_ft.forward(Variable(images))

        outputs = outputs.view(-1, outputs.numel())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('correct = %d ...' % correct)
    print('total = %d ...' % total)
    print('Accuracy of the network on the test images: %f' % (correct / total))


def finetune_anet(model_ft, train_loader, test_loader, criterion, num_epochs=25, inference=False):
    """
    fine-tune ANet from pre-trained VGG-M Face model
    :param model_ft:
    :param train_loader:
    :param test_loader:
    :param criterion:
    :param num_epochs:
    :param inference:
    :return:
    """
    num_ftrs = model_ft.fc8.in_channels
    model_ft.fc8 = nn.Conv2d(num_ftrs, 1, 1)

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model_ft = nn.DataParallel(model_ft)

    if torch.cuda.is_available():
        model_ft = model_ft.cuda()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            exp_lr_scheduler.step()
            model_ft.train(True)

            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, labels = data['image'], data['score']

                # wrap them in Variable
                if torch.cuda.is_available():
                    model_ft = model_ft.cuda()
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.float().cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels.float())

                # zero the parameter gradients
                optimizer_ft.zero_grad()

                # forward + backward + optimize
                outputs = model_ft(inputs)
                outputs = outputs.view(-1, 2)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer_ft.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f' %
                          (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(model_ft.state_dict(), os.path.join(model_path_dir, 'anet.pth'))

    else:
        print('Loading pre-trained model...')
        model_ft.load_state_dict(torch.load(os.path.join('./model/anet.pth')))

    model_ft.train(False)
    predicted_labels = []
    gt_labels = []

    for data in test_loader:
        images, labels = data['image'], data['score']
        if torch.cuda.is_available():
            model_ft = model_ft.cuda()
            labels = labels.cuda()
            outputs = model_ft.forward(Variable(images.cuda()))
        else:
            outputs = model_ft.forward(Variable(images))

        predicted_labels += outputs.cpu().data.numpy().tolist()
        gt_labels += labels.cpu().numpy().tolist()

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(mean_absolute_error(np.array(gt_labels), np.array(predicted_labels).ravel()), 4)
    rmse_lr = round(np.math.sqrt(mean_squared_error(np.array(gt_labels), np.array(predicted_labels).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_labels), np.array(predicted_labels).ravel())[0, 1], 4)

    print('===============The Mean Absolute Error of ANet is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of ANet is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of ANet is {0}===================='.format(pc))


def train_hmtnet(hmt_net, train_loader, test_loader, num_epochs=25, inference=False):
    """
    train HMT-Net
    :param hmt_net:
    :param train_loader:
    :param test_loader:
    :param num_epochs:
    :param inference:
    :return:
    """
    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        hmt_net = nn.DataParallel(hmt_net)

    if torch.cuda.is_available():
        hmt_net = hmt_net.cuda()

    criterion = HMTLoss()
    optimizer = optim.SGD(hmt_net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    if not inference:
        for epoch in range(num_epochs):  # loop over the dataset multiple times

            exp_lr_scheduler.step()
            hmt_net.train(True)

            running_loss = 0.0

            for i, data in enumerate(train_loader, 0):
                # get the inputs
                inputs, gender, race, attractiveness = data['image'], data['gender'], data['race'], \
                                                       data['attractiveness']

                # wrap them in Variable
                if torch.cuda.is_available():
                    hmt_net = hmt_net.cuda()
                    inputs, gender, race, attractiveness = Variable(inputs.cuda()), Variable(gender.cuda()), Variable(
                        race.cuda()), Variable(attractiveness.float().cuda())
                else:
                    inputs, gender, race, attractiveness = Variable(inputs), Variable(gender), Variable(race), \
                                                           Variable(attractiveness.float())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                g_pred, r_pred, a_pred = hmt_net.forward(inputs)
                # print('g_pred.shape = ' + str(g_pred.shape))
                # print('r_pred.shape = ' + str(r_pred.shape))
                # print('a_pred.shape = ' + str(a_pred.shape))

                g_pred = g_pred.view(cfg['batch_size'], 2)
                r_pred = r_pred.view(cfg['batch_size'], 2)
                a_pred = a_pred.view(cfg['batch_size'], 1)

                loss = criterion(g_pred, gender, r_pred, race, a_pred, attractiveness)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 100 == 99:  # print every 100 mini-batches
                    print('[%d, %5d] loss: %.5f' % (epoch + 1, i + 1, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')
        print('Save trained model...')

        model_path_dir = './model'
        file_utils.mkdirs_if_not_exist(model_path_dir)
        torch.save(hmt_net.state_dict(), os.path.join(model_path_dir, 'hmt-net.pth'))

    else:
        print('Loading pre-trained model...')
        hmt_net.load_state_dict(torch.load(os.path.join('./model/hmt-net.pth')))

    hmt_net.train(False)

    predicted_attractiveness_values = []
    gt_attractiveness_values = []

    total = 0
    g_correct = 0
    r_correct = 0

    for data in test_loader:
        images, g_gt, r_gt, a_gt = data['image'], data['gender'], data['race'], \
                                   data['attractiveness']
        if torch.cuda.is_available():
            hmt_net = hmt_net.cuda()
            g_gt = g_gt.cuda()
            r_gt = r_gt.cuda()
            a_gt = a_gt.cuda()

            g_pred, r_pred, a_pred = hmt_net.forward(Variable(images.cuda()))
        else:
            g_pred, r_pred, a_pred = hmt_net.forward(Variable(images))

        predicted_attractiveness_values += a_pred.cpu().data.numpy().tolist()
        gt_attractiveness_values += a_gt.cpu().numpy().tolist()

        g_pred = g_pred.view(-1, g_pred.numel())
        r_pred = r_pred.view(-1, r_pred.numel())
        _, g_predicted = torch.max(g_pred.data, 1)
        _, r_predicted = torch.max(r_pred.data, 1)
        total += g_gt.size(0)
        g_correct += (g_predicted == g_gt).sum()
        r_correct += (r_predicted == r_gt).sum()

    print('total = %d ...' % total)
    print('Gender correct sample = %d ...' % g_correct)
    print('Race correct sample = %d ...' % r_correct)
    print('Accuracy of Race Classification: %f' % (r_correct / total))
    print('Accuracy of Gender Classification: %f' % (g_correct / total))

    from sklearn.metrics import mean_absolute_error, mean_squared_error

    mae_lr = round(
        mean_absolute_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel()), 4)
    rmse_lr = round(np.math.sqrt(
        mean_squared_error(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())), 4)
    pc = round(np.corrcoef(np.array(gt_attractiveness_values), np.array(predicted_attractiveness_values).ravel())[0, 1],
               4)

    print('===============The Mean Absolute Error of HMT-Net is {0}===================='.format(mae_lr))
    print('===============The Root Mean Square Error of HMT-Net is {0}===================='.format(rmse_lr))
    print('===============The Pearson Correlation of HMT-Net is {0}===================='.format(pc))


if __name__ == '__main__':
    # gnet = GNet()
    # rnet = RNet()

    # vgg_m_face = vgg_m_face_bn_dag.load_vgg_m_face_bn_dag()
    data_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.ColorJitter(),
        transforms.Normalize(mean=[131.45376586914062, 103.98748016357422, 91.46234893798828],
                             std=[1, 1, 1])
    ])

    # hand-crafted train and test loader for gender data set
    # male_shuffled_indices = np.random.permutation(2750)
    # female_shuffled_indices = np.random.permutation(2750)
    # train_loader = torch.utils.data.DataLoader(
    #     FaceGenderDataset(transform=data_transform, male_shuffled_indices=male_shuffled_indices,
    #                       female_shuffled_indices=female_shuffled_indices, train=True),
    #     batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(FaceGenderDataset(transform=data_transform,
    #                                                             male_shuffled_indices=male_shuffled_indices,
    #                                                             female_shuffled_indices=female_shuffled_indices,
    #                                                             train=False), batch_size=cfg['batch_size'],
    #                                           shuffle=False, num_workers=4)

    # hand-crafted train and test loader for race data set
    # yellow_shuffled_indices = np.random.permutation(4000)
    # white_shuffled_indices = np.random.permutation(1500)
    # train_loader = torch.utils.data.DataLoader(
    #     FaceRaceDataset(transform=data_transform, yellow_shuffled_indices=yellow_shuffled_indices,
    #                     white_shuffled_indices=white_shuffled_indices, train=True),
    #     batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(FaceRaceDataset(transform=data_transform,
    #                                                           yellow_shuffled_indices=yellow_shuffled_indices,
    #                                                           white_shuffled_indices=white_shuffled_indices,
    #                                                           train=False), batch_size=cfg['batch_size'],
    #                                           shuffle=False, num_workers=4)

    # gender_dataset = datasets.ImageFolder(root=cfg['gender_base_dir'],
    #                                       transform=data_transform)
    # race_dataset = datasets.ImageFolder(root=cfg['race_base_dir'],
    #                                     transform=data_transform)
    # train_loader, test_loader = data_loader.split_train_and_test_with_py_datasets(data_set=gender_dataset,
    #                                                                               batch_size=cfg['batch_size'])

    # criterion = nn.CrossEntropyLoss()

    # print('***************************start training GNet***************************')
    # optimizer = optim.SGD(gnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # train_gnet(gnet, train_loader, test_loader, criterion, optimizer, num_epochs=2, inference=False)
    # print('***************************finish training GNet***************************')

    # print('***************************start training RNet***************************')
    # optimizer = optim.SGD(rnet.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    # train_rnet(rnet, train_loader, test_loader, criterion, optimizer, num_epochs=2, inference=False)
    # print('***************************finish training RNet***************************')

    # print('***************************start fine-tuning VGGMFace***************************')
    # finetune_vgg_m_model(vgg_m_face, train_loader, test_loader, criterion, 2, False)
    # print('***************************finish fine-tuning VGGMFace***************************')

    # print('---------------------------------------------------------------------------')

    # train_loader = torch.utils.data.DataLoader(data_loader.FBPDataset(True, transform=data_transform),
    #                                            batch_size=cfg['batch_size'], shuffle=True, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(data_loader.FBPDataset(False, transform=data_transform),
    #                                           batch_size=cfg['batch_size'], shuffle=False, num_workers=4)
    #
    # print('***************************start fine-tuning ANet***************************')
    # finetune_anet(vgg_m_face, train_loader, test_loader, nn.MSELoss(), 2, False)

    print('+++++++++++++++++++++++++++++++++++++++++start training HMT-Net+++++++++++++++++++++++++++++++++++++++++')
    hmtnet = HMTNet()
    # hand-crafted train and test loader for face data set
    train_loader = torch.utils.data.DataLoader(FaceDataset(cv_index=1, train=True, transform=data_transform),
                                               batch_size=cfg['batch_size'], shuffle=True, num_workers=4,
                                               drop_last=True)
    test_loader = torch.utils.data.DataLoader(FaceDataset(cv_index=1, train=False, transform=data_transform),
                                              batch_size=cfg['batch_size'], shuffle=False, num_workers=4,
                                              drop_last=True)

    train_hmtnet(hmtnet, train_loader, test_loader, 2, False)
    print('+++++++++++++++++++++++++++++++++++++++++finish training HMT-Net+++++++++++++++++++++++++++++++++++++++++')
