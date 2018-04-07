import sys
import os

import numpy as np
import pandas as pd
from skimage import io
from PIL import Image
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

    def __init__(self, csv_file=cfg['SCUT_FBP5500_csv'], root_dir=cfg['gender_base_dir'], transform=None,
                 male_shuffled_indices=None, female_shuffled_indices=None, train=True):
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

        tmp = get_fileindex_and_label()
        m_label = [tmp[_][0] for _ in m_fileindex_list]
        f_label = [tmp[_][0] for _ in f_fileindex_list]

        male_train_set_size = int(len(m_fileindex_list) * 0.6)
        female_train_set_size = int(len(f_fileindex_list) * 0.6)

        male_train_indices = male_shuffled_indices[:male_train_set_size]
        male_test_indices = male_shuffled_indices[male_train_set_size:]
        female_train_indices = female_shuffled_indices[:female_train_set_size]
        female_test_indices = female_shuffled_indices[female_train_set_size:]

        if train:
            self.image_files = pd.concat(
                [pd.DataFrame(m_fileindex_list).iloc[male_train_indices],
                 pd.DataFrame(f_fileindex_list).iloc[female_train_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(m_label).iloc[male_train_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(f_label).iloc[female_train_indices].values.ravel().tolist())])
        else:
            self.image_files = pd.concat(
                [pd.DataFrame(m_fileindex_list).iloc[male_test_indices],
                 pd.DataFrame(f_fileindex_list).iloc[female_test_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(m_label).iloc[male_test_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(f_label).iloc[female_test_indices].values.ravel().tolist())])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.image_labels.values.ravel().tolist()[idx]
        img_name = os.path.join(self.root_dir, 'M' if label == 1 else 'F', self.image_files.values.tolist()[idx][0])
        image = io.imread(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FaceRaceDataset(Dataset):
    """
    Face Race dataset with hierarchical sampling strategy
    """

    def __init__(self, csv_file=cfg['SCUT_FBP5500_csv'], root_dir=cfg['race_base_dir'], transform=None,
                 yellow_shuffled_indices=None, white_shuffled_indices=None, train=True):
        self.root_dir = root_dir
        self.img_index = pd.read_csv(csv_file, header=None, sep=',').iloc[:, 2]
        self.img_label = pd.DataFrame(np.array([1 if _ == 'w' else 0 for _ in
                                                pd.read_csv(csv_file, header=None, sep=',').iloc[:,
                                                1].values.tolist()]).ravel())

        def get_fileindex_and_label():
            fileindex_and_label = {}
            for i in range(len(self.img_index.tolist())):
                fileindex_and_label[self.img_index.values.tolist()[i]] = self.img_label.values.tolist()[i]

            return fileindex_and_label

        y_fileindex_list = os.listdir(os.path.join(cfg['race_base_dir'], 'Y'))
        w_fileindex_list = os.listdir(os.path.join(cfg['race_base_dir'], 'W'))

        tmp = get_fileindex_and_label()
        y_label = [tmp[_][0] for _ in y_fileindex_list]
        w_label = [tmp[_][0] for _ in w_fileindex_list]

        yellow_train_set_size = int(len(y_fileindex_list) * 0.6)
        white_train_set_size = int(len(w_fileindex_list) * 0.6)

        yellow_train_indices = yellow_shuffled_indices[:yellow_train_set_size]
        yellow_test_indices = yellow_shuffled_indices[yellow_train_set_size:]
        white_train_indices = white_shuffled_indices[:white_train_set_size]
        white_test_indices = white_shuffled_indices[white_train_set_size:]

        if train:
            self.image_files = pd.concat(
                [pd.DataFrame(y_fileindex_list).iloc[yellow_train_indices],
                 pd.DataFrame(w_fileindex_list).iloc[white_train_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(y_label).iloc[yellow_train_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(w_label).iloc[white_train_indices].values.ravel().tolist())])
        else:
            self.image_files = pd.concat(
                [pd.DataFrame(y_fileindex_list).iloc[yellow_test_indices],
                 pd.DataFrame(w_fileindex_list).iloc[white_test_indices]])

            self.image_labels = pd.concat(
                [pd.DataFrame(pd.DataFrame(y_label).iloc[yellow_test_indices].values.ravel().tolist()),
                 pd.DataFrame(pd.DataFrame(w_label).iloc[white_test_indices].values.ravel().tolist())])

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.image_labels.values.ravel().tolist()[idx]
        img_name = os.path.join(self.root_dir, 'W' if label == 1 else 'Y', self.image_files.values.tolist()[idx][0])
        image = io.imread(img_name)
        sample = {'image': image, 'label': label}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

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
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class FaceDataset(Dataset):
    """
    Face Dataset for SCUT-FBP5500
    """

    def __init__(self, cv_index=1, train=True, transform=None):
        if train:
            self.face_img = pd.read_csv(
                os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index, 'train_%d.txt' % cv_index),
                sep=' ', header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index,
                                                       'train_%d.txt' % cv_index),
                                          sep=' ', header=None).iloc[:, 1].astype(np.float).tolist()
        else:
            self.face_img = pd.read_csv(
                os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index, 'test_%d.txt' % cv_index),
                sep=' ',
                header=None).iloc[:, 0].tolist()
            self.face_score = pd.read_csv(os.path.join(cfg['cv_split_base_dir'], 'cross_validation_%d' % cv_index,
                                                       'test_%d.txt' % cv_index), sep=' ', header=None).iloc[:, 1] \
                .astype(np.float).tolist()

        self.transform = transform

    def __len__(self):
        return len(self.face_img)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scutfbp5500_images_dir'], self.face_img[idx]))
        attractiveness = self.face_score[idx]
        gender = 1 if self.face_img[idx].split('.')[0][0] == 'm' else 0
        race = 1 if self.face_img[idx].split('.')[0][2] == 'w' else 0

        sample = {'image': image, 'attractiveness': attractiveness, 'gender': gender, 'race': race}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample


class ScutFBP(Dataset):
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
        image = io.imread(os.path.join(cfg['scutfbp_images_dir'], self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
