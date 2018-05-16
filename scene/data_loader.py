import json
import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from scene.cfg import cfg


class IndoorDataset(Dataset):
    """
    Indoor CVPR2009 dataset
    """

    def __init__(self, train=True, transform=None):
        mapping = json.load(open('./mapping.json', mode='rt'))
        print(mapping)

        img_file_list = []
        img_label_list = []
        img_category_list = []

        if train:
            with open(os.path.join(cfg['base_dir'], 'TrainImages.txt'), mode='rt') as f:
                for line in f.readlines():
                    # img_file_list.append(line.strip().replace('\n', ''))
                    img_file_list.append(line.split('/')[1].strip().replace('\n', ''))
                    img_label_list.append(mapping[line.split('/')[0]])
                    img_category_list.append(line.split('/')[0])
        else:
            with open(os.path.join(cfg['base_dir'], 'TestImages.txt'), mode='rt') as f:
                for line in f.readlines():
                    # img_file_list.append(line.strip().replace('\n', ''))
                    img_file_list.append(line.split('/')[1].strip().replace('\n', ''))
                    img_label_list.append(mapping[line.split('/')[0]])
                    img_category_list.append(line.split('/')[0])

        self.files = img_file_list
        self.labels = img_label_list
        self.categories = img_category_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # image = io.imread(os.path.join(cfg['base_dir'], 'Images', self.files[idx]))
        print(os.path.join(cfg['base_dir'], 'Images', self.categories[idx], self.files[idx]))
        image = io.imread(os.path.join(cfg['base_dir'], 'Images', self.categories[idx], self.files[idx]))
        cls = self.labels[idx]

        sample = {'image': image, 'class': cls}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
