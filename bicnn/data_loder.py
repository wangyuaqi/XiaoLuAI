import os
import sys
import math
from collections import OrderedDict

import numpy as np
import pandas as pd
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from bicnn.cfg import cfg


class ScutFBPDataset(Dataset):
    """
    SCUT-FBP dataset
    """

    def __init__(self, f_list, f_labels, transform=None):
        self.face_files = f_list
        self.face_score = f_labels.tolist()
        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], 'SCUT-FBP-%d.jpg' % self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class ScutFBPExpDataset(Dataset):
    """
    SCUT-FBP Expected dataset
    """

    def __init__(self, f_list, f_labels, transform=None):
        self.face_files = f_list
        self.face_score = [round(_) for _ in f_labels.tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], 'SCUT-FBP-%d.jpg' % self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class HotOrNotDataset(Dataset):
    def __init__(self, cv_split=1, train=True, transform=None):
        df = pd.read_csv(
            os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0], 'eccv2010_split%d.csv' % cv_split),
            header=None)

        filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for
                     _ in df.iloc[:, 0].tolist()]
        scores = df.iloc[:, 1].tolist()
        flags = df.iloc[:, 2].tolist()

        train_set = OrderedDict()
        test_set = OrderedDict()

        for i in range(len(flags)):
            if flags[i] == 'train':
                train_set[filenames[i]] = scores[i]
            elif flags[i] == 'test':
                test_set[filenames[i]] = scores[i]

        if train:
            self.face_files = list(train_set.keys())
            self.face_scores = list(train_set.values())
        else:
            self.face_files = list(test_set.keys())
            self.face_scores = list(test_set.values())

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        score = self.face_scores[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class HotOrNotExpDataset(Dataset):
    def __init__(self, cv_split=1, train=True, transform=None):
        df = pd.read_csv(
            os.path.join(os.path.split(os.path.abspath(cfg['hotornot_dir']))[0], 'eccv2010_split%d.csv' % cv_split),
            header=None)

        filenames = [os.path.join(cfg['hotornot_dir'], _.replace('.bmp', '.jpg')) for
                     _ in df.iloc[:, 0].tolist()]
        scores = df.iloc[:, 1].tolist()
        flags = df.iloc[:, 2].tolist()

        train_set = OrderedDict()
        test_set = OrderedDict()

        for i in range(len(flags)):
            if flags[i] == 'train':
                train_set[filenames[i]] = round(scores[i]) + 3
            elif flags[i] == 'test':
                test_set[filenames[i]] = round(scores[i]) + 3

        if train:
            self.face_files = list(train_set.keys())
            self.face_scores = list(train_set.values())
        else:
            self.face_files = list(test_set.keys())
            self.face_scores = list(test_set.values())

        self.transform = transform

    def __len__(self):
        return len(self.face_files)

    def __getitem__(self, idx):
        image = io.imread(self.face_files[idx])
        score = self.face_scores[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
