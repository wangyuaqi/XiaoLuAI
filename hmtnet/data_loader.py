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
    Face Gender dataset with hierarchical sampling strategy
    """

    def __init__(self, csv_file=cfg['SCUT_FBP5500_csv'], root_dir=cfg['gender_base_dir'], transform=None):
        self.root_dir = root_dir
        self.img_index = pd.read_csv(csv_file, header=None, sep=',').iloc[:, 2]
        self.img_label = pd.DataFrame(np.array([1 if _ == 'm' else 0 for _ in
                                                pd.read_csv(csv_file, header=None, sep=',').iloc[:,
                                                0].values.tolist()]).ravel())

        def get_fileindex_and_label():
            fileindex_and_label = {}
            for i in range(len(self.img_index.tolist())):
                fileindex_and_label[self.img_index.values.tolist()[i]] = self.img_label.values.tolist()[i]

            return fileindex_and_label

        m_fileindex_list = os.listdir(os.path.join(cfg['gender_base_dir'], 'M'))
        f_fileindex_list = os.listdir(os.path.join(cfg['gender_base_dir'], 'F'))

        male_shuffled_indices = np.random.permutation(len(m_fileindex_list))
        female_shuffled_indices = np.random.permutation(len(f_fileindex_list))

        male_train_set_size = int(len(m_fileindex_list) * 0.6)
        male_train_indices = male_shuffled_indices[:male_train_set_size]
        female_train_set_size = int(len(f_fileindex_list) * 0.6)
        female_train_indices = female_shuffled_indices[:female_train_set_size]

        self.training_set = pd.concat(
            [self.img_index.iloc[male_train_indices], self.img_index.iloc[female_train_indices]])
        self.training_labels = pd.concat(
            [self.img_label.iloc[male_train_indices], self.img_label.iloc[female_train_indices]])

        male_test_indices = male_shuffled_indices[male_train_set_size:]
        female_test_indices = female_shuffled_indices[female_train_set_size:]

        self.test_set = pd.concat(
            [pd.DataFrame(m_fileindex_list).iloc[male_test_indices],
             pd.DataFrame(f_fileindex_list).iloc[female_test_indices]])

        m_label = [get_fileindex_and_label()[_] for _ in m_fileindex_list]
        f_label = [get_fileindex_and_label()[_] for _ in f_fileindex_list]

        self.test_labels = pd.concat(
            [pd.DataFrame(m_label).iloc[male_test_indices], pd.DataFrame(f_label).iloc[female_test_indices]])

        self.transform = transform

    def __len__(self):
        return len(self.img_index)

    def __getitem__(self, idx):
        label = self.img_label.values.tolist()[idx]
        img_name = os.path.join(self.root_dir, 'M' if label == 1 else 'F', self.img_index.values.tolist()[idx])
        image = io.imread(img_name)
        sample = {'image': image, 'label': label}

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
