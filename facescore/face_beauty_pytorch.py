"""
face beauty prediction implemented by PyTorch
"""
import os
import sys
import warnings

import cv2
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
from facescore.config import *


class FBNet(nn.Module):
    """network definition"""

    def __init__(self):
        super(FBNet, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(128 * 9 * 9, 512, nn.Dropout(0.5))
        self.fc2 = nn.Linear(512, 1, nn.Dropout(0.5))

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 3)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class FaceBeautyDataset(Dataset):
    """Face Beauty Dataset"""

    def __init__(self, train=True, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        train_filenames, train_scores, test_filenames, test_scores = split_data(config['test_ratio'])

        if train:
            self.face_images = train_filenames
            self.beauty_labels = train_scores
            self.transform = transform
        else:
            self.face_images = test_filenames
            self.beauty_labels = test_scores
            self.transform = transform

    def __len__(self):
        return len(self.face_images)

    def __getitem__(self, idx):
        img_name = self.face_images[idx]
        image = cv2.imread(img_name[0])
        face_score = self.beauty_labels[idx]

        sample = {'image': np.transpose(image - np.mean(image), [2, 0, 1]), 'score': face_score}

        if self.transform:
            sample = self.transform(sample['image'])

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or tuple): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, face_score = sample['image'], sample['score']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = cv2.resize(image, (new_h, new_w))

        return {'image': img, 'score': face_score}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, face_score = sample['image'], sample['score']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                left: left + new_w]

        return {'image': image, 'score': face_score}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, face_score = sample['image'], sample['score']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'score': torch.from_numpy(face_score)}


def split_data(test_ratio):
    """
    split dataset into train set and test set
    :return:
    """
    df = pd.read_excel(config['label_excel_path'], 'Sheet1')

    filename_indexs = pd.DataFrame([config['face_image_filename'].format(_) for _ in df['Image'].tolist()])
    attractiveness_scores = df['Attractiveness label']

    shuffled_indices = np.random.permutation(len(df))
    test_set_size = int(len(df) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]

    return filename_indexs.iloc[train_indices].values, attractiveness_scores.iloc[train_indices].values, \
           filename_indexs.iloc[test_indices].values, attractiveness_scores.iloc[test_indices].values


def train():
    """
    train FBNet model
    :return:
    """
    if not os.path.exists('./model/') or not os.path.isdir('./model/'):
        os.makedirs('./model/')

    print('Start training')
    transform = transforms.Compose(
        [transforms.RandomCrop(config['image_size']),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transformed_dataset = FaceBeautyDataset(transform=None)
    train_dataloader = DataLoader(transformed_dataset, batch_size=config['batch_size'],
                                  shuffle=True, num_workers=4)
    test_dataloader = DataLoader(FaceBeautyDataset(train=False), batch_size=config['batch_size'],
                                 shuffle=True, num_workers=4)

    net = FBNet().double()
    if torch.cuda.is_available():
        net = net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-5)

    for epoch in range(config['epoch']):  # loop over the dataset multiple times

        running_loss = 0.0
        for i_batch, sample_batched in enumerate(train_dataloader):
            inputs, labels = sample_batched['image'], sample_batched['score']
            if torch.cuda.is_available():
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            criterion.zero_grad()
            outputs = net.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i_batch % 200 == 0:
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i_batch + 1, running_loss / 2000))
                running_loss = 0.0

                print("Save model...")
                torch.save(net.state_dict(), './model/fbnet.pth')
                print('Model has been saved...')

    print('Finished Training.\nStart evaluating...')


if __name__ == '__main__':
    train()

    net = FBNet().double().cuda()
    net.load_state_dict(torch.load('model/fbnet.pth'))
    test_dataloader = DataLoader(FaceBeautyDataset(train=False), batch_size=config['batch_size'],
                                 shuffle=True, num_workers=4)
    predicted_output = []
    gt = []
    for data in test_dataloader:
        inputs, labels = data['image'], data['score']
        if torch.cuda.is_available():
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        outputs = net(inputs)
        gt += labels.cpu().data.numpy().tolist()
        predicted_output += outputs.cpu().data.numpy().tolist()

    pc = stats.pearsonr(np.array(gt).ravel(), np.array(predicted_output).ravel())
    mae = mean_absolute_error(np.array(gt).ravel(), np.array(predicted_output).ravel())
    rmse = np.sqrt(mean_squared_error(np.array(gt).ravel(), np.array(predicted_output).ravel()))
    print('=' * 100)
    print('Pearson Correlation is ' + str(pc))
    print('=' * 100)
    print('MAE is ' + str(mae))
    print('=' * 100)
    print('RMSE is ' + str(rmse))
    print('=' * 100)
