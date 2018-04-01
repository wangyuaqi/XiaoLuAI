import sys
import os

import numpy as np
import pandas as pd
from skimage import io, transform
import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

sys.path.append('../')
from hmtnet.cfg import cfg


def split_train_and_test_with_py_datasets(data_set, batch_size=cfg['batch_size'], test_size=0.2, num_works=4,
                                          pin_memory=True):
    """
    split datasets into train and test loader
    :param data_set:
    :param batch_size:
    :param test_size:
    :param num_works:
    :param pin_memory:
    :return:
    """
    num_dataset = len(data_set)
    indices = list(range(num_dataset))
    split = int(np.floor(test_size * num_dataset))

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=data_set, batch_size=batch_size, sampler=test_sampler, num_workers=num_works,
        pin_memory=pin_memory
    )

    return train_loader, test_loader


class FaceGenderDataset(Dataset):
    """
    Face Gender dataset
    """

    def __init__(self, X, y, transform=None):
        self.images = X
        self.labels = y
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {'image': self.images.iloc[idx - 1].as_matrix().astype(np.float32),
                  'label': self.labels.iloc[idx - 1].as_matrix().astype(np.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class FBPDataset(Dataset):
    """
    SCUT-FBP5500 dataset
    """

    def __init__(self, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'train.txt'), sep=' ', header=None).iloc[:,
                            0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'train.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'test.txt'), sep=' ', header=None).iloc[:,
                            0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['4_6_split_dir'], 'test.txt'), sep=' ', header=None).iloc[:,
                              1].astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_images_dir'], self.face_img[idx]))
        score = self.face_score[idx]
        sample = {'image': image, 'score': score}

        if self.transform:
            from PIL import Image
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
