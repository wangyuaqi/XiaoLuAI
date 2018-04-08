import os
import sys
import time
from pprint import pprint

from skimage.transform import resize
from skimage import io
from PIL import Image
import cv2
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../')
from hmtnet.cfg import cfg

from hmtnet.models import HMTNet


def inference(image_file, hmtnet_model_file='./model/hmt-net.pth'):
    """
    inference with pre-trained HMT-Net
    :param image_file:
    :param hmtnet_model_file:
    :return:
    """
    hmt_net = HMTNet()
    hmt_net.load_state_dict(torch.load(hmtnet_model_file))

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)

    image = resize(io.imread(image_file), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])

    input = torch.from_numpy(image).unsqueeze(0).float()

    if torch.cuda.is_available():
        hmt_net = hmt_net.cuda()
        input = Variable(input.cuda())

    hmt_net.train(False)

    tik = time.time()
    g_pred, r_pred, a_pred = hmt_net.forward(input)
    tok = time.time()

    g_pred = g_pred.view(1, 2)
    r_pred = r_pred.view(1, 2)
    a_pred = a_pred.view(1, 1)

    _, g_predicted = torch.max(g_pred.data, 1)
    _, r_predicted = torch.max(r_pred.data, 1)

    g_pred = 'male' if int(g_predicted.cpu()) == 1 else 'female'
    r_pred = 'white' if int(r_predicted.cpu()) == 1 else 'yellow'

    return {'image': os.path.basename(image_file), 'gender': g_pred, 'race': r_pred,
            'attractiveness': float(a_pred.cpu()), 'elapse': tok - tik}


def feature_viz(image_file, hmtnet_model_file='./model/hmt-net.pth'):
    hmt_net = HMTNet()
    hmt_net.load_state_dict(torch.load(hmtnet_model_file))

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)

    image = resize(io.imread(image_file), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()

    if torch.cuda.is_available():
        hmt_net = hmt_net.cuda()
        input = Variable(input.cuda())

    for idx, module in hmt_net.named_children():
        if idx != 'relu3':
            input = module(input)
        else:
            mat = np.transpose(input[0, 0:3, :, :].data.cpu().numpy(), [1, 2, 0])
            cv2.imshow('relu3', mat)
            cv2.imwrite('./relu3.jpg', mat)
            cv2.waitKey()
            cv2.destroyAllWindows()
            break


def cal_elapse(nn_name, img_file):
    import torchvision.models as models

    image = resize(io.imread(img_file), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()

    if nn_name == 'AlexNet':
        alex_net = models.AlexNet(num_classes=1)
        if torch.cuda.is_available():
            input = Variable(input.cuda())
            alex_net = alex_net.cuda()

        start = time.time()
        alex_net.forward(input)
        end = time.time()

        return end - start

    elif nn_name == 'ResNet18':
        resnet18 = models.resnet18(num_classes=1)
        if torch.cuda.is_available():
            input = Variable(input.cuda())
            resnet18 = resnet18.cuda()

        start = time.time()
        resnet18.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet50':
        resnet50 = models.resnet50(num_classes=1)
        if torch.cuda.is_available():
            input = Variable(input.cuda())
            resnet50 = resnet50.cuda()

        start = time.time()
        resnet50.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet101':
        resnet101 = models.resnet101(num_classes=1)
        if torch.cuda.is_available():
            input = Variable(input.cuda())
            resnet101 = resnet101.cuda()

        start = time.time()
        resnet101.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet152':
        resnet152 = models.resnet152(num_classes=1)
        if torch.cuda.is_available():
            input = Variable(input.cuda())
            resnet152 = resnet152.cuda()

        start = time.time()
        resnet152.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'HMT-Net':
        hmt_net = HMTNet()
        hmt_net.load_state_dict(torch.load('./model/hmt-net.pth'))

        if torch.cuda.is_available():
            input = Variable(input.cuda())
            hmt_net = hmt_net.cuda()

        start = time.time()
        hmt_net.forward(input)
        end = time.time()

        return end - start
    else:
        print('Invalid NN Name!!')
        sys.exit(0)


if __name__ == '__main__':
    result_list = []
    for img_file in os.listdir(cfg['scutfbp5500_images_dir']):
        result = inference(os.path.join(cfg['scutfbp5500_images_dir'], img_file))
        print(result)
        result_list.append([result['image'], result['attractiveness'], result['gender'][0], result['race'][0]])

        col = ['image', 'pred_attractiveness', 'pred_gender', 'pred_race']
        df = pd.DataFrame(result_list, columns=col)
        df.to_excel("./results.xlsx", sheet_name='Results', index=False)

        # cv2.imshow('image', cv2.imread(os.path.join(cfg['scutfbp5500_images_dir'], img_file)))
        # cv2.waitKey()
        # cv2.destroyAllWindows()

    # feature_viz(os.path.join(cfg['scutfbp5500_images_dir'], 'ftw8.jpg'))

    # print(cal_elapse('AlexNet', '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images/ftw8.jpg'))
    # print(inference('/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images/ftw8.jpg'))
