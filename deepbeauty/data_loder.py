import os
import sys

import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

sys.path.append('../')
from deepbeauty.cfg import cfg


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
        image = io.imread(os.path.join(cfg['scut_fbp_dir'], self.face_files[idx]))
        score = self.face_score[idx]

        sample = {'image': image, 'score': score}

        if self.transform:
            sample['image'] = self.transform(Image.fromarray(sample['image'].astype(np.uint8)))

        return sample
