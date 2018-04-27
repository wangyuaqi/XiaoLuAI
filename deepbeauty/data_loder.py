import os
import sys

import numpy as np
import pandas as pd
from skimage import io, transform
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

sys.path.append('../')
from deepbeauty.cfg import cfg


class ScutFBP(Dataset):
    """
    SCUT-FBP dataset
    """

    def __init__(self, csv_file='./cvsplit/SCUT-FBP.xlsx', transform=None):
        df = pd.read_excel(csv_file, sheet_name='Sheet1', header=True)
        self.img_indices = df['Image'].tolist()
        self.face_scores = df['Attractiveness label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.img_indices)

    def __getitem__(self, idx):
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], self.img_indices[idx]))
        score = self.face_scores[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
