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

import visdom
import torch
import torch.nn as nn
from torch.autograd import Variable

sys.path.append('../')
from hmtnet.cfg import cfg

from hmtnet.models import HMTNet


def inference(img, hmtnet_model_file='./model/hmt-net.pth'):
    """
    inference with pre-trained HMT-Net
    :param image_file: an image filepath or image numpy array
    :param hmtnet_model_file:
    :return:
    """
    hmt_net = HMTNet()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hmt_net.load_state_dict(torch.load(hmtnet_model_file))

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)

    if type(img) is str:
        image = resize(io.imread(img), (224, 224), mode='constant')
    else:
        img = cv2.resize(img, (224, 224))
        image = img.astype(np.float64)

    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])

    input = torch.from_numpy(image).unsqueeze(0).float()

    hmt_net = hmt_net.to(device)
    input = input.to(device)

    hmt_net.eval()

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

    return {'gender': g_pred, 'race': r_pred, 'attractiveness': float(a_pred.cpu()), 'elapse': tok - tik}


def feature_viz(image_file, hmtnet_model_file='./model/hmt-net.pth'):
    """
    feature visualization
    :param image_file:
    :param hmtnet_model_file:
    :return:
    """
    hmt_net = HMTNet()
    hmt_net.load_state_dict(torch.load(hmtnet_model_file))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1:
        print("We are running on", torch.cuda.device_count(), "GPUs!")
        hmt_net = nn.DataParallel(hmt_net)

    image = resize(io.imread(image_file), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()

    hmt_net = hmt_net.to(device)
    input = input.to(device)

    for idx, module in hmt_net.named_children():
        if idx != 'conv4':
            input = module(input)
        else:
            vis = visdom.Visdom()
            # vis.image(np.transpose(input.data.cpu().numpy(), [1, 2, 0]))

            mat = np.transpose(input[0, 10:13, :, :].data.cpu().numpy(), [1, 2, 0])
            mat = cv2.resize(mat, (128, 128))
            cv2.imshow('conv4', mat)
            print(mat.shape)
            import scipy.misc

            if not os.path.isdir('./feature_viz/'):
                os.makedirs('./feature_viz/')

            scipy.misc.imsave('./feature_viz/' + os.path.basename(image_file).split('.')[0] + '-conv4.jpg', mat)
            cv2.waitKey()
            cv2.destroyAllWindows()
            break


def cal_elapse(nn_name, img_file):
    """
    calculate time elapse
    :param nn_name:
    :param img_file:
    :return:
    """
    import torchvision.models as models

    image = resize(io.imread(img_file), (224, 224), mode='constant')
    image[:, :, 0] -= 131.45376586914062
    image[:, :, 1] -= 103.98748016357422
    image[:, :, 2] -= 91.46234893798828

    image = np.transpose(image, [2, 0, 1])
    input = torch.from_numpy(image).unsqueeze(0).float()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if nn_name == 'AlexNet':
        alex_net = models.AlexNet(num_classes=1)
        input = input.to(device)
        alex_net = alex_net.to(device)

        start = time.time()
        alex_net.forward(input)
        end = time.time()

        return end - start

    elif nn_name == 'ResNet18':
        resnet18 = models.resnet18(num_classes=1)
        input = input.to(device)
        resnet18 = resnet18.to(device)

        start = time.time()
        resnet18.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet50':
        resnet50 = models.resnet50(num_classes=1)
        input = input.to(device)
        resnet50 = resnet50.to(device)

        start = time.time()
        resnet50.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet101':
        resnet101 = models.resnet101(num_classes=1)
        input = input.to(device)
        resnet101 = resnet101.to(device)

        start = time.time()
        resnet101.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'ResNet152':
        resnet152 = models.resnet152(num_classes=1)
        input = input.to(device)
        resnet152 = resnet152.to(device)

        start = time.time()
        resnet152.forward(input)
        end = time.time()

        return end - start
    elif nn_name == 'HMT-Net':
        hmt_net = HMTNet()
        hmt_net.load_state_dict(torch.load('./model/hmt-net.pth'))

        input = input.to(device)
        hmt_net = hmt_net.to(device)

        start = time.time()
        hmt_net.forward(input)
        end = time.time()

        return end - start
    else:
        print('Invalid NN Name!!')
        sys.exit(0)


def output_result(is_show=True):
    """
    output result on test set
    :param is_show:
    :return:
    """
    result_list = []

    df = pd.read_csv(os.path.join(cfg['cv_split_base_dir'], 'cross_validation_1', 'test_1.txt'), sep=' ', header=None)
    img_list = df.iloc[:, 0].tolist()
    score_list = df.iloc[:, 1].tolist()

    for i in range(len(img_list)):
        print('Processing image : %s, its gt score is %f' % (img_list[i], score_list[i]))

        result = inference(os.path.join(cfg['scutfbp5500_images_dir'], img_list[i]))
        print(result)
        result_list.append(
            [img_list[i], result['attractiveness'], score_list[i], result['gender'][0],
             img_list[i][0], result['race'][0],
             img_list[i].split('.')[0][2]])

        col = ['image', 'pred_attractiveness', 'gt_attractiveness', 'pred_gender', 'gt_gender', 'pred_race', 'gt_race']
        df = pd.DataFrame(result_list, columns=col)
        df.to_excel("./results.xlsx", sheet_name='Results', index=False)

        if is_show:
            cv2.imshow('image', cv2.imread(os.path.join(cfg['scutfbp5500_images_dir'], img_list[i])))
            cv2.waitKey()
            cv2.destroyAllWindows()


def infer_and_show_img(img_filepath):
    """
    inference with image contains only one person
    :param img_filepath:
    :return:
    """
    from hmtnet.img_utils import det_landmarks
    dlib_result = det_landmarks(img_filepath)
    print(dlib_result)
    hmt_result = inference(img_filepath)
    print(hmt_result)

    image = cv2.imread(img_filepath)

    if image.shape[0] >= image.shape[1]:
        image = image[0:image.shape[1], :, :]
    else:
        image = image[:, 0:image.shape[0], :]
    # image = cv2.resize(image, (400, 400))

    h, w, c = image.shape
    text_area_height = 50
    final_image = np.zeros([h + text_area_height, 2 * w, c], dtype=np.uint8)
    final_image[0:h, 0:w, :] = image
    final_image[h:h + text_area_height, :, :] = 255

    face = dlib_result[0]
    if hmt_result['gender'] == 'male':
        color = (255, 0, 0)
    else:
        color = (0, 0, 255)

    race_text = 'Asian' if hmt_result['race'] == 'yellow' else 'Westerner'

    cv2.rectangle(image, (face['bbox'][0], face['bbox'][1]), (face['bbox'][2], face['bbox'][3]), color, 2)
    cv2.putText(final_image,
                'Face Beauty Value:{0}   Race:{1}   Gender:{2}'.format(str(round(hmt_result['attractiveness'], 2)),
                                                                       race_text, hmt_result['gender']),
                (int(w / 10), h + int(text_area_height / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99, 99, 238), 0,
                cv2.LINE_AA)

    for ldmk in face['landmarks']:
        cv2.circle(image, (ldmk[0], ldmk[1]), 2, (255, 245, 0), -1)

    final_image[0:h, w:2 * w, :] = image

    cv2.imwrite('./final_image.jpg', final_image)
    cv2.imshow('final_image', final_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


def infer_and_show_mul_people_img(img_filepath):
    """
    inference with image contains multiple faces
    :param img_filepath:
    :return:
    """
    from hmtnet.img_utils import det_landmarks
    dlib_result = det_landmarks(img_filepath)
    print(dlib_result)

    image = cv2.imread(img_filepath)

    if image.shape[0] >= image.shape[1]:
        image = image[0:image.shape[1], :, :]
    else:
        image = image[:, 0:image.shape[0], :]
    # image = cv2.resize(image, (400, 400))

    h, w, c = image.shape
    text_area_height = 50
    final_image = np.zeros([h + text_area_height, 2 * w, c], dtype=np.uint8)
    final_image[0:h, 0:w, :] = image
    final_image[h:h + text_area_height, :, :] = 255

    for k, v in dlib_result.items():
        face = dlib_result[k]
        hmt_result = inference(
            cv2.imread(img_filepath)[face['bbox'][0]: face['bbox'][2], face['bbox'][1]: face['bbox'][3]])
        print(hmt_result)
        if hmt_result['gender'] == 'male':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)

        race_text = 'Asian' if hmt_result['race'] == 'yellow' else 'Westerner'

        cv2.rectangle(image, (face['bbox'][0], face['bbox'][1]), (face['bbox'][2], face['bbox'][3]), color, 2)
        cv2.putText(image, str(round(hmt_result['attractiveness'], 2)), (face['bbox'][0], face['bbox'][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 0, cv2.LINE_AA)

        for ldmk in face['landmarks']:
            cv2.circle(image, (ldmk[0], ldmk[1]), 2, (255, 245, 0), -1)

        final_image[0:h, w:2 * w, :] = image

    cv2.putText(final_image, 'AI Technology Supported by @LucasX', (int(w / 4), h + int(text_area_height / 2)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (99, 99, 238), 0, cv2.LINE_AA)
    cv2.imwrite('./final_image.jpg', final_image)
    cv2.imshow('final_image', final_image)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # feature_viz(os.path.join(cfg['scutfbp5500_images_dir'], 'fty688.jpg'))

    infer_and_show_mul_people_img('./psb.jpg')

    # print(inference('/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images/ftw8.jpg'))
    # print(cal_elapse('AlexNet', '/media/lucasx/Document/DataSet/Face/SCUT-FBP5500/Images/ftw8.jpg'))
