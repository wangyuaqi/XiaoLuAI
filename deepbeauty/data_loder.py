import os
import sys

import pandas as pd
from skimage import io, transform
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from deepbeauty.cfg import cfg


class ScutFBPDataset(Dataset):
    """
    SCUT-FBP dataset.
    """

    def __init__(self, csv_file, transform=None):
        self.landmarks_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(cfg['scut_fbp_dir'],
                                self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        sample = {'image': image, 'score': landmarks}

        if self.transform:
            sample = self.transform(sample)

        return sample
